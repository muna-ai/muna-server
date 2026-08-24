/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use muna::beta::openai::{ImageCreateParams, ImageSize};
use muna::types::Acceleration;
use serde::Deserialize;

use crate::handlers::error::AppError;
use crate::serving::predict;
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct ImageGenerationsRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    n: Option<i32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    background: Option<String>,
    #[serde(default)]
    output_compression: Option<i32>,
}

/// OpenAI-compatible image generations via muna-rs `images.generate`.
pub(crate) async fn image_generations(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ImageGenerationsRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let size = parse_size(req.size.as_deref())?;
    let model = state.registry.ensure_ready(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let params = ImageCreateParams {
        prompt: req.prompt,
        model: req.model,
        background: req.background,
        n: req.n,
        output_format: req.output_format,
        output_compression: req.output_compression,
        size,
        acceleration: Some(Acceleration::LocalGpu),
    };
    let muna = model.muna.clone();
    let response = predict::run(move || async move {
        muna.beta.openai.images.generate(params).await
    }).await?;
    drop(guard);
    Ok(Json(response).into_response())
}

fn parse_size(value: Option<&str>) -> Result<Option<ImageSize>, AppError> {
    match value {
        None | Some("auto") => Ok(None),
        Some("256x256") => Ok(Some(ImageSize::Size256x256)),
        Some("512x512") => Ok(Some(ImageSize::Size512x512)),
        Some("1024x1024") => Ok(Some(ImageSize::Size1024x1024)),
        Some("1536x1024") => Ok(Some(ImageSize::Size1536x1024)),
        Some("1024x1536") => Ok(Some(ImageSize::Size1024x1536)),
        Some("1792x1024") => Ok(Some(ImageSize::Size1792x1024)),
        Some("1024x1792") => Ok(Some(ImageSize::Size1024x1792)),
        Some(value) => Err(AppError::bad_request(format!(
            "unsupported size `{value}`"
        ))),
    }
}
