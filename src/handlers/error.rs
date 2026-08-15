/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::serving::registry::RegistryError;

/// Uniform OpenAI-style error envelope for every handler.
pub(crate) struct AppError {
    status: StatusCode,
    body: Value,
    /// `Retry-After` seconds, set for 503 responses.
    retry_after: Option<u64>,
}

impl AppError {
    pub(crate) fn bad_request(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                }
            }),
            retry_after: None,
        }
    }

    pub(crate) fn unavailable(message: String, retry_after: u64) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: json!({
                "error": {
                    "message": message,
                    "type": "unavailable_error",
                }
            }),
            retry_after: Some(retry_after),
        }
    }

    pub(crate) fn internal(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: json!({
                "error": {
                    "message": message,
                    "type": "muna_error",
                }
            }),
            retry_after: None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body)).into_response();
        if let Some(retry_after) = self.retry_after {
            if let Ok(value) = retry_after.to_string().parse() {
                response.headers_mut().insert("Retry-After", value);
            }
        }
        response
    }
}

impl From<muna::MunaError> for AppError {
    fn from(e: muna::MunaError) -> Self {
        Self::internal(e.to_string())
    }
}

impl From<RegistryError> for AppError {
    fn from(e: RegistryError) -> Self {
        match e {
            RegistryError::Loading { retry_after } => Self::unavailable(
                "model is loading".into(),
                retry_after
            ),
            RegistryError::Failed(error) => Self::internal(
                format!("model failed to load: {error}")
            ),
        }
    }
}

/// OpenAI-style error payload for embedding in an SSE stream.
pub(crate) fn muna_error_value(e: &muna::MunaError) -> Value {
    json!({
        "error": {
            "message": e.to_string(),
            "type": "muna_error",
        }
    })
}
