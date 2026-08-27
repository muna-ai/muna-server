/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::serving::registry::RegistryError;

/// `axum::Json` with an OpenAI-shaped rejection: a malformed body yields a
/// 400 `{"error": {...}}` envelope instead of axum's plain-text 422 (OpenAI
/// itself uses 400 for malformed bodies, and clients only render the
/// envelope).
pub(crate) struct Json<T>(pub T);

impl<S, T> FromRequest<S> for Json<T>
where
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(AppError::bad_request(rejection.body_text())),
        }
    }
}

/// Response-side parity with `axum::Json`, so handlers need only the one
/// import for both extraction and JSON responses.
impl<T: serde::Serialize> IntoResponse for Json<T> {

    fn into_response(self) -> Response {
        axum::Json(self.0).into_response()
    }
}

/// [`Json`] with the rejection re-shaped to Anthropic's error envelope,
/// for `/v1/messages`.
pub(crate) struct AnthropicJson<T>(pub T);

impl<S, T> FromRequest<S> for AnthropicJson<T>
where
    axum::Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = AnthropicError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(
                AnthropicError::from(AppError::bad_request(rejection.body_text()))
            ),
        }
    }
}

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
        let mut response = (self.status, axum::Json(self.body)).into_response();
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

/// Anthropic-style error envelope: same statuses and messages as
/// [`AppError`], re-shaped to `{"type": "error", "error": {"type", "message"}}`.
pub(crate) struct AnthropicError(AppError);

impl IntoResponse for AnthropicError {

    fn into_response(self) -> Response {
        let AppError { status, body, retry_after } = self.0;
        let message = body["error"]["message"]
            .as_str()
            .unwrap_or("unknown error")
            .to_string();
        let body = json!({
            "type": "error",
            "error": {
                "type": anthropic_error_type(status),
                "message": message,
            }
        });
        let mut response = (status, axum::Json(body)).into_response();
        if let Some(retry_after) = retry_after {
            if let Ok(value) = retry_after.to_string().parse() {
                response.headers_mut().insert("Retry-After", value);
            }
        }
        response
    }
}

/// Re-shape any [`AppError`] into the Anthropic envelope, so handlers can
/// build errors with the existing constructors and convert with `.into()`.
impl From<AppError> for AnthropicError {

    fn from(e: AppError) -> Self {
        Self(e)
    }
}

/// Let `?` lift muna prediction errors into 500 `api_error` responses.
impl From<muna::MunaError> for AnthropicError {

    fn from(e: muna::MunaError) -> Self {
        Self(AppError::from(e))
    }
}

/// Let `?` lift registry errors (loading, failed, not served) into their
/// respective 429 / 500 / 404 Anthropic-shaped responses.
impl From<RegistryError> for AnthropicError {

    fn from(e: RegistryError) -> Self {
        Self(AppError::from(e))
    }
}

fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST         => "invalid_request_error",
        StatusCode::NOT_FOUND           => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS   => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _                               => "api_error",
    }
}

/// Anthropic-style error payload for embedding in an SSE stream.
pub(crate) fn anthropic_error_value(e: &muna::MunaError) -> Value {
    json!({
        "type": "error",
        "error": {
            "message": e.to_string(),
            "type": "api_error",
        }
    })
}
