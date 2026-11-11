/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! OpenAI Images editing accepts multipart uploads and a single text prompt.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, State};
use axum::response::{IntoResponse, Response};
use muna::beta::openai::ImageEditParams;

use crate::handlers::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail};
use crate::state::AppState;

pub(crate) async fn image_edits(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let multipart = multipart.map_err(|e| AppError::bad_request(e.to_string()))?;
    let mut params = parse(multipart).await?;
    let model = state.inference_model(&params.model).await?;
    params.devices = model.device_handles()?;
    params.acceleration = Some(model.acceleration());
    state.check_in_if_due(&params.model).await;
    state.mark_model_loaded(params.model.clone()).await;
    let admitted = Instant::now();
    let _admission = model.admit().await?;
    let guard = state.dispatcher.acquire(&params.model, &model).await;
    let queue_wait = admitted.elapsed();
    let muna = model.muna.clone();
    let dispatched = Instant::now();
    let response = predict::run_model(model.clone(), guard, move || async move {
        muna.beta.openai.images.edit(params).await
    })
    .await?;
    let images = response.data.as_ref().map_or(0, |data| data.len()) as u32;
    model.stats.telemetry.record(PredictionSample {
        at: Instant::now(),
        queue_wait,
        latency: dispatched.elapsed(),
        detail: SampleDetail::Image { images },
    });
    Ok(Json(response).into_response())
}

async fn parse(mut multipart: Multipart) -> Result<ImageEditParams, AppError> {
    let mut params = ImageEditParams::default();
    let mut seen = HashSet::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::multipart(e.status(), e.body_text()))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        if matches!(name.as_str(), "image" | "image[]") {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| AppError::multipart(e.status(), e.body_text()))?;
            if bytes.is_empty() {
                return Err(AppError::bad_request("Image file is empty.".into()));
            }
            params.image.push(bytes.to_vec().into());
            continue;
        }
        if !seen.insert(name.clone()) {
            return Err(AppError::bad_request(format!(
                "Duplicate image edit field `{name}`."
            )));
        }
        if !matches!(
            name.as_str(),
            "model"
                | "prompt"
                | "n"
                | "size"
                | "output_format"
                | "output_compression"
                | "background"
                | "response_format"
        ) {
            return Err(AppError::bad_request(format!(
                "Unsupported image edit field `{name}`."
            )));
        }
        let value = field
            .text()
            .await
            .map_err(|e| AppError::multipart(e.status(), e.body_text()))?;
        match name.as_str() {
            "model" => {
                if value.trim().is_empty() {
                    return Err(AppError::bad_request("The model must not be empty.".into()));
                }
                params.model = if !value.starts_with('@') && value.contains('/') {
                    format!("@{value}")
                } else {
                    value
                };
            }
            "prompt" => params.prompt = value,
            "n" => {
                let n = value
                    .parse::<i32>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or_else(|| AppError::bad_request("Image count must be positive.".into()))?;
                params.n = Some(n);
            }
            "size" => params.size = super::images::parse_size(Some(&value))?,
            "output_format" => {
                if !matches!(value.as_str(), "png" | "jpeg" | "webp") {
                    return Err(AppError::bad_request(
                        "Unsupported image output format.".into(),
                    ));
                }
                params.output_format = Some(value);
            }
            "output_compression" => {
                let quality = value
                    .parse::<i32>()
                    .ok()
                    .filter(|q| (0..=100).contains(q))
                    .ok_or_else(|| {
                        AppError::bad_request(
                            "Output compression must be between 0 and 100.".into(),
                        )
                    })?;
                params.output_compression = Some(quality);
            }
            "background" => {
                if !matches!(value.as_str(), "auto" | "opaque" | "transparent") {
                    return Err(AppError::bad_request(
                        "Unsupported image background.".into(),
                    ));
                }
                params.background = Some(value);
            }
            "response_format" if value != "b64_json" => {
                return Err(AppError::bad_request(
                    "Only b64_json image responses are supported.".into(),
                ));
            }
            _ => {}
        }
    }
    if !seen.contains("model") || !seen.contains("prompt") || params.image.is_empty() {
        return Err(AppError::bad_request(
            "Image editing requires `model`, `prompt`, and `image` fields.".into(),
        ));
    }
    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::{FromRequest, Request};
    use muna::beta::openai::ImageInput;

    async fn upload(fields: &[(&str, &str)]) -> Multipart {
        let mut body = String::new();
        for (name, value) in fields {
            body.push_str(&format!("--test-boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"));
        }
        body.push_str("--test-boundary--\r\n");
        let request = Request::builder()
            .header(
                "content-type",
                "multipart/form-data; boundary=test-boundary",
            )
            .body(Body::from(body))
            .unwrap();
        Multipart::from_request(request, &()).await.unwrap()
    }

    #[tokio::test]
    async fn parses_scalar_prompt_and_repeated_reference_images() {
        let params = parse(
            upload(&[
                ("model", "owner/albedo"),
                ("prompt", ""),
                ("image[]", "first"),
                ("image[]", "second"),
                ("n", "2"),
                ("size", "512x512"),
                ("output_format", "png"),
                ("response_format", "b64_json"),
            ])
            .await,
        )
        .await
        .unwrap_or_else(|_| panic!("Valid image edit multipart failed to parse"));
        assert_eq!(params.model, "@owner/albedo");
        assert_eq!(params.prompt, "");
        assert_eq!(params.n, Some(2));
        assert_eq!(params.size.unwrap().dimensions(), (Some(512), Some(512)));
        assert!(matches!(&params.image[0], ImageInput::Encoded(v) if v == b"first"));
        assert!(matches!(&params.image[1], ImageInput::Encoded(v) if v == b"second"));
    }

    #[tokio::test]
    async fn rejects_missing_duplicate_and_unsupported_fields() {
        for extra in [
            ("prompt", "duplicate"),
            ("n", "0"),
            ("n", "garbage"),
            ("output_compression", "101"),
            ("output_format", "invalid"),
            ("mask", "unsupported"),
            ("devices", "0"),
            ("response_format", "url"),
            ("image", ""),
        ] {
            let fields = [
                ("model", "@owner/albedo"),
                ("prompt", "ignored"),
                ("image", "file"),
                extra,
            ];
            assert!(
                parse(upload(&fields).await).await.is_err(),
                "accepted {extra:?}"
            );
        }
        for fields in [
            vec![("model", "@owner/albedo"), ("image", "file")],
            vec![("model", "@owner/albedo"), ("prompt", "")],
            vec![("prompt", ""), ("image", "file")],
        ] {
            assert!(parse(upload(&fields).await).await.is_err());
        }
    }
}
