/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! HTTP surface, one module per route (API families grouped by protocol):
//!
//! - [`health`], [`status`], [`drain`]: operational endpoints consumed by supervisors and the control plane.
//! - [`predictions`]: Muna-native prediction endpoint.
//! - [`openai`]: OpenAI-compatible API.
//! - [`anthropic`]: Anthropic-compatible API.
//! - [`not_found`]: fallback for unknown routes.
//! - [`error`]: OpenAI- and Anthropic-style error envelopes shared by the API handlers.
//! - [`body`]: chat request body wrapper shared by the OpenAI and Anthropic handlers.

mod anthropic;
mod body;
mod drain;
mod error;
mod health;
mod not_found;
mod openai;
mod predictions;
mod status;

use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderName, Method, header};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};

use crate::state::AppState;

/// Request body limit for inference routes, which carry inline base64
/// images. Twice Anthropic's 32 MB Messages API limit; other routes keep
/// axum's 2 MB default.
pub(crate) const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

/// The complete route table. State is applied by the caller.
pub(crate) fn router() -> Router<Arc<AppState>> {
    let limit = || DefaultBodyLimit::max(MAX_REQUEST_BYTES);
    Router::new()
        // Health and management
        .route("/", get(health::health))
        .route("/health", get(health::health))
        .route("/status", get(status::status))
        .route("/drain", post(drain::drain))
        // Muna remote prediction
        .route("/v1/predictions/remote", post(predictions::predictions).layer(limit()))
        // OpenAI compatibility
        .route("/v1/models", get(openai::models))
        .route("/v1/models/{*id}", get(openai::model))
        .route("/v1/chat/completions", post(openai::chat_completions).layer(limit()))
        .route("/v1/embeddings", post(openai::embeddings).layer(limit()))
        .route("/v1/images/generations", post(openai::image_generations).layer(limit()))
        // Anthropic compatibility
        .route("/v1/messages", post(anthropic::messages).layer(limit()))
        // Fallbacks
        .fallback(not_found::not_found)
        // Browser clients need CORS when a node is used standalone via
        // `muna deploy` (mirrors OpenAI's `access-control-allow-origin: *`).
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers([
                    header::AUTHORIZATION,
                    header::CONTENT_TYPE,
                    HeaderName::from_static("x-api-key"),
                    HeaderName::from_static("anthropic-version")
                ])
                .expose_headers([header::RETRY_AFTER])
        )
}
