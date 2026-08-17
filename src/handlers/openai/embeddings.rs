/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use muna::beta::openai::EncodingFormat;
use muna::types::Acceleration;
use serde::Deserialize;

use crate::handlers::error::AppError;
use crate::serving::predict;
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct EmbeddingsRequest {
    input: EmbeddingsInput,
    model: String,
    dimensions: Option<i32>,
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingsInput {
    String(String),
    Strings(Vec<String>),
}

impl EmbeddingsInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::String(input) => vec![input],
            Self::Strings(input) => input,
        }
    }
}

pub(crate) async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let input = req.input.into_vec();
    let encoding_format = parse_encoding_format(req.encoding_format.as_deref())?;
    let model = state.registry.ensure_ready(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let muna = state.muna.clone();
    let tag = req.model;
    let dimensions = req.dimensions;
    let response = predict::run(move || async move {
        muna.beta.openai.embeddings.create(
            input,
            &tag,
            dimensions,
            encoding_format,
            Some(Acceleration::LocalGpu)
        ).await
    }).await?;
    drop(guard);
    Ok(Json(response).into_response())
}

fn parse_encoding_format(value: Option<&str>) -> Result<Option<EncodingFormat>, AppError> {
    match value {
        Some("float") => Ok(Some(EncodingFormat::Float)),
        Some("base64") => Ok(Some(EncodingFormat::Base64)),
        Some(value) => Err(AppError::bad_request(format!(
            "unsupported encoding_format `{value}`"
        ))),
        None => Ok(None),
    }
}
