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
    /// `Retry-After` seconds, set for 429 (loading) and 503 (draining).
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

    /// 503: the node itself cannot serve (draining).
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

    /// 429 with a `Retry-After` header: the model exists but cannot serve
    /// yet (loading). Mirrors OpenAI's rate-limit error shape so clients
    /// with standard backoff handling retry automatically.
    pub(crate) fn retry_later(message: String, retry_after: u64) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: json!({
                "error": {
                    "message": message,
                    "type": "rate_limit_error",
                }
            }),
            retry_after: Some(retry_after),
        }
    }

    /// 404 with OpenAI's `model_not_found` error shape: the tag is not in
    /// this server's pinned model set (`--models`).
    pub(crate) fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: json!({
                "error": {
                    "message": format!("model '{model}' is not served by this deployment"),
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }),
            retry_after: None,
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
            RegistryError::Loading { retry_after } => Self::retry_later(
                "model is loading".into(),
                retry_after
            ),
            RegistryError::Failed(error) => Self::internal(
                format!("model failed to load: {error}")
            ),
            RegistryError::NotServed(model) => Self::model_not_found(&model),
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
