/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! OpenAI Images editing with JSON references or multipart image uploads.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{FromRequest, Multipart, Request, State};
use axum::response::{IntoResponse, Response};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use muna::beta::adapters::{
    Adapter, DepthFormat, OpenAIDepthOptions, OpenAIImageOptions,
    OpenAIModel3DOptions,
};
use muna::beta::openai::ImageEditParams;

use crate::handlers::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail};
use crate::state::AppState;

pub(crate) async fn image_edits(
    State(state): State<Arc<AppState>>,
    request: ImageEditsRequest,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let (mut params, urls) = request.into_params()?;
    let model = state.inference_model(&params.model).await?;
    let depth = model
        .signature
        .outputs
        .iter()
        .any(|p| p.denotation.as_deref() == Some("depth_map"));
    let model3d = !depth
        && model
            .signature
            .outputs
            .iter()
            .any(|p| p.denotation.as_deref() == Some("3d"));
    let depth_adapter = depth.then(|| model.adapters.depth()).transpose()?;
    let model3d_adapter = model3d.then(|| model.adapters.model3d()).transpose()?;
    let image_adapter = (!depth && !model3d)
        .then(|| model.adapters.edits())
        .transpose()?;
    let encoding = if !model3d {
        Some(depth_options(depth, &params)?)
    } else {
        None
    };
    let model3d_encoding = if model3d {
        Some(model3d_options(&params)?)
    } else {
        None
    };
    params.image = state
        .image_fetcher
        .resolve_images(urls)
        .await
        .map_err(AppError::bad_request)?
        .into_iter()
        .map(Into::into)
        .collect();
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
        let (width, height) = params
            .size
            .unwrap_or(muna::beta::openai::ImageSize::Auto)
            .dimensions();
        let inputs = if let Some(adapter) = &depth_adapter {
            adapter.inputs(params.image, params.prompt, params.n, width, height)?
        } else if let Some(adapter) = &model3d_adapter {
            adapter.inputs(params.image, params.prompt, params.n, width, height)?
        } else {
            image_adapter
                .as_ref()
                .expect("image adapter selected")
                .inputs(params.image, params.prompt, params.n, width, height)?
        };
        let prediction = muna
            .predictions
            .create(
                &params.model,
                Some(inputs),
                params.acceleration,
                params.devices.as_deref(),
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        if let Some(adapter) = depth_adapter {
            adapter.adapt(prediction, encoding).await
        } else if let Some(adapter) = model3d_adapter {
            adapter.adapt(prediction, model3d_encoding).await
        } else {
            image_adapter
                .expect("image adapter selected")
                .adapt(
                    prediction,
                    Some(OpenAIImageOptions {
                        output_format: params.output_format.unwrap_or_else(|| "png".into()),
                        output_compression: params.output_compression,
                    }),
                )
                .await
        }
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
            .is_some_and(|s| !matches!(s, "png" | "jpeg" | "webp" | "npy" | "npz" | "ply"))
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
            prompt: Some(self.prompt),
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

// Keep the JSON URL-reference surface while accepting the multipart body
// produced by OpenAI's image= file argument. Existing fetch limits apply to
// uploaded images too, after converting them to data URL references.
impl<S: Send + Sync> FromRequest<S> for ImageEditsRequest {
    type Rejection = AppError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let multipart = request
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value
                    .split(';')
                    .next()
                    .is_some_and(|value| value.trim().eq_ignore_ascii_case("multipart/form-data"))
            });
        if !multipart {
            return Json::<Self>::from_request(request, state)
                .await
                .map(|Json(value)| value);
        }
        let mut multipart = Multipart::from_request(request, state)
            .await
            .map_err(|error| bad(&error.to_string()))?;
        let mut fields = serde_json::Map::new();
        let mut images = Vec::new();
        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| AppError::multipart(error.status(), error.body_text()))?
        {
            let name = field.name().unwrap_or_default().to_owned();
            if matches!(name.as_str(), "image" | "image[]") {
                if field.file_name().is_none() {
                    return Err(bad("Multipart image fields must contain uploaded files."));
                }
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|error| AppError::multipart(error.status(), error.body_text()))?;
                if bytes.is_empty() || images.len() >= 1024 {
                    return Err(bad(
                        "Image files must be nonempty; at most 1024 are supported.",
                    ));
                }
                images.push(serde_json::json!({
                    "image_url": format!("data:image/png;base64,{}", BASE64.encode(bytes))
                }));
                continue;
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
                return Err(bad(&format!("Unsupported image edit field `{name}`.")));
            }
            if fields.contains_key(&name) || field.file_name().is_some() {
                return Err(bad(&format!(
                    "Duplicate or invalid image edit field `{name}`."
                )));
            }
            let value = field
                .text()
                .await
                .map_err(|error| AppError::multipart(error.status(), error.body_text()))?;
            let value = if matches!(name.as_str(), "n" | "output_compression") {
                serde_json::Value::from(
                    value
                        .parse::<i32>()
                        .map_err(|_| bad("Image count/compression must be integers."))?,
                )
            } else {
                serde_json::Value::String(value)
            };
            fields.insert(name, value);
        }
        fields.insert("images".into(), serde_json::Value::Array(images));
        serde_json::from_value(serde_json::Value::Object(fields))
            .map_err(|error| bad(&error.to_string()))
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
    fn ply_assets_use_the_3d_response_path_and_reject_image_encoding_options() {
        let body = json!({
            "model":"@test/idol", "prompt":"", "output_format":"ply",
            "images":[{"image_url":"https://example.com/input.png"}]
        });
        let request: ImageEditsRequest = serde_json::from_value(body).unwrap();
        let (mut params, _) = request.into_params().ok().unwrap();
        assert_eq!(model3d_options(&params).ok().unwrap().format, "ply");
        assert!(depth_options(false, &params).is_err());
        params.output_format = Some("png".into());
        assert!(model3d_options(&params).is_err());
        params.output_format = None;
        assert_eq!(model3d_options(&params).ok().unwrap().format, "ply");
        params.output_compression = Some(90);
        assert!(model3d_options(&params).is_err());
        params.output_compression = None;
        params.background = Some("auto".into());
        assert!(model3d_options(&params).is_err());
    }

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
        assert_eq!(params.prompt.as_deref(), Some(""));
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
    async fn accepts_sdk_multipart_upload_and_rejects_duplicate_scalars() {
        let body = concat!(
            "--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nowner/depth\r\n",
            "--test\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n\r\n",
            "--test\r\nContent-Disposition: form-data; name=\"output_format\"\r\n\r\nply\r\n",
            "--test\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\nimage bytes\r\n",
            "--test--\r\n"
        );
        let request = Request::builder()
            .header("content-type", "multipart/form-data; boundary=test")
            .body(Body::from(body))
            .unwrap();
        let parsed = ImageEditsRequest::from_request(request, &())
            .await
            .ok()
            .unwrap();
        let (params, images) = parsed.into_params().ok().unwrap();
        assert_eq!(params.output_format.as_deref(), Some("ply"));
        assert_eq!(images.len(), 1);
        let duplicate = body.replace(
            "--test--",
            "--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nother\r\n--test--",
        );
        let request = Request::builder()
            .header("content-type", "multipart/form-data; boundary=test")
            .body(Body::from(duplicate))
            .unwrap();
        assert!(ImageEditsRequest::from_request(request, &()).await.is_err());
    }
}

// Output format belongs to the HTTP response, never to predictor arguments.
fn depth_options(depth: bool, params: &ImageEditParams) -> Result<OpenAIDepthOptions, AppError> {
    let format = match params.output_format.as_deref() {
        Some("npy") if depth => DepthFormat::Npy,
        Some("npz") if depth => DepthFormat::Npz,
        _ if depth => return Err(bad("Depth output requires output_format=npy or npz.")),
        Some("ply") => return Err(bad("PLY output requires a binary output annotated as 3d.")),
        Some("npy" | "npz") => {
            return Err(bad(
                "NumPy output requires an array_list annotated as depth_map.",
            ))
        }
        _ => return Ok(OpenAIDepthOptions::default()),
    };
    if params.background.is_some() || params.output_compression.is_some() {
        return Err(bad(
            "Depth output does not accept background or output_compression.",
        ));
    }
    Ok(OpenAIDepthOptions { format })
}

fn model3d_options(params: &ImageEditParams) -> Result<OpenAIModel3DOptions, AppError> {
    if params.background.is_some() || params.output_compression.is_some() {
        return Err(bad(
            "3D output does not accept background or output_compression.",
        ));
    }
    let options = OpenAIModel3DOptions {
        format: params.output_format.clone().unwrap_or_else(|| "ply".into()),
    };
    options.validate()?;
    Ok(options)
}
