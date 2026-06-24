/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use muna::beta::openai::EncodingFormat;
use muna::types::Acceleration;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::AppState;

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
    let model = req.model;
    let input = req.input.into_vec();
    let encoding_format = parse_encoding_format(req.encoding_format.as_deref())?;

    let _guard = state.predict_lock.clone().lock_owned().await;
    let response = state
        .muna
        .beta
        .openai
        .embeddings
        .create(
            input,
            &model,
            req.dimensions,
            encoding_format,
            Some(Acceleration::LocalGpu),
        )
        .await?;
    mark_model_loaded(&state, model).await;

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

async fn mark_model_loaded(state: &AppState, model: String) {
    let mut loaded_models = state.loaded_models.write().await;
    loaded_models.entry(model).or_insert_with(now);
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) struct AppError {
    status: StatusCode,
    body: Value,
}

impl AppError {
    fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                }
            }),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

impl From<muna::MunaError> for AppError {
    fn from(e: muna::MunaError) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: json!({
                "error": {
                    "message": e.to_string(),
                    "type": "muna_error",
                }
            }),
        }
    }
}
