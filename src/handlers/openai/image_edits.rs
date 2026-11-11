/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! JSON-only OpenAI Images editing with scalar prompts and image URL references.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use muna::beta::openai::ImageEditParams;

use crate::handlers::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail};
use crate::state::AppState;

pub(crate) async fn image_edits(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ImageEditsRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let (mut params, urls) = request.into_params()?;
    params.image = state
        .image_fetcher
        .resolve_images(urls)
        .await
        .map_err(AppError::bad_request)?
        .into_iter()
        .map(Into::into)
        .collect();
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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ImageEditsRequest {
    model: String,
    prompt: String,
    images: Vec<ImageReference>,
    n: Option<i32>,
    size: Option<String>,
    output_format: Option<String>,
    output_compression: Option<i32>,
    background: Option<String>,
    response_format: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageReference {
    image_url: String,
}

impl ImageEditsRequest {

    fn into_params(self) -> Result<(ImageEditParams, Vec<String>), AppError> {
        if self.model.trim().is_empty() {
            return Err(bad("The model must not be empty."));
        }
        if self.images.is_empty()
            || self
                .images
                .iter()
                .any(|image| image.image_url.trim().is_empty())
        {
            return Err(bad("Image editing requires a nonempty `images` array of `image_url` references. File IDs are not supported."));
        }
        if self.n.is_some_and(|n| n <= 0) {
            return Err(bad("Image count must be positive."));
        }
        if self
            .output_compression
            .is_some_and(|q| !(0..=100).contains(&q))
        {
            return Err(bad("Output compression must be between 0 and 100."));
        }
        if self
            .output_format
            .as_deref()
            .is_some_and(|s| !matches!(s, "png" | "jpeg" | "webp"))
        {
            return Err(bad("Unsupported image output format."));
        }
        if self
            .background
            .as_deref()
            .is_some_and(|s| !matches!(s, "auto" | "opaque" | "transparent"))
        {
            return Err(bad("Unsupported image background."));
        }
        if self
            .response_format
            .as_deref()
            .is_some_and(|s| s != "b64_json")
        {
            return Err(bad("Only b64_json image responses are supported."));
        }
        let params = ImageEditParams {
            model: if !self.model.starts_with('@') && self.model.contains('/') {
                format!("@{}", self.model)
            } else {
                self.model
            },
            prompt: self.prompt,
            n: self.n,
            size: super::images::parse_size(self.size.as_deref())?,
            output_format: self.output_format,
            output_compression: self.output_compression,
            background: self.background,
            ..Default::default()
        };
        Ok((
            params,
            self.images
                .into_iter()
                .map(|image| image.image_url)
                .collect(),
        ))
    }
}

fn bad(message: &str) -> AppError {
    AppError::bad_request(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::{FromRequest, Request};
    use serde_json::json;

    #[test]
    fn parses_scalar_prompt_and_multiple_image_references() {
        let request: ImageEditsRequest = serde_json::from_value(json!({
            "model": "owner/albedo", "prompt": "", "n": 2, "size": "512x512",
            "response_format": "b64_json", "images": [
                {"image_url": "data:image/png;base64,Zmlyc3Q="},
                {"image_url": "https://example.com/second.png"}
            ]
        }))
        .unwrap();
        let (params, urls) = request
            .into_params()
            .unwrap_or_else(|_| panic!("valid request"));
        assert_eq!(params.model, "@owner/albedo");
        assert_eq!(params.prompt, "");
        assert_eq!(params.n, Some(2));
        assert_eq!(params.size.unwrap().dimensions(), (Some(512), Some(512)));
        assert_eq!(
            urls,
            [
                "data:image/png;base64,Zmlyc3Q=",
                "https://example.com/second.png"
            ]
        );
    }

    #[test]
    fn rejects_invalid_and_unsupported_fields() {
        let valid = json!({"model":"@owner/albedo", "prompt":"", "images":[{"image_url":"https://example.com/a.png"}]});
        for (field, value) in [
            ("prompt", json!(["batched"])),
            ("n", json!(0)),
            ("n", json!("2")),
            ("output_compression", json!(101)),
            ("output_format", json!("invalid")),
            ("mask", json!({})),
            ("devices", json!([0])),
            ("response_format", json!("url")),
            ("images", json!([])),
            ("images", json!([{"file_id":"file-123"}])),
            ("images", json!([{"image_url":""}])),
            ("image", json!("legacy")),
        ] {
            let mut body = valid.clone();
            body[field] = value;
            let accepted = serde_json::from_value::<ImageEditsRequest>(body)
                .ok()
                .is_some_and(|req| req.into_params().is_ok());
            assert!(!accepted, "accepted invalid {field}");
        }
        for field in ["model", "prompt", "images"] {
            let mut body = valid.clone();
            body.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<ImageEditsRequest>(body).is_err());
        }
        assert!(serde_json::from_str::<ImageEditsRequest>(
            r#"{"model":"a","model":"b","prompt":"","images":[]}"#
        )
        .is_err());
    }

    #[tokio::test]
    async fn rejects_multipart_body() {
        let request = Request::builder()
            .header("content-type", "multipart/form-data; boundary=test")
            .body(Body::from("--test--\r\n"))
            .unwrap();
        assert!(axum::Json::<ImageEditsRequest>::from_request(request, &())
            .await
            .is_err());
    }
}
