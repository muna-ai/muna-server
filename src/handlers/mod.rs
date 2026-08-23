/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! HTTP surface, grouped by protocol family:
//!
//! - [`ops`]: liveness, node status, drain, fallback consumed by supervisors and the control plane.
//! - [`predictions`]: Muna-native prediction endpoint.
//! - [`openai`]: OpenAI-compatible API.
//! - [`anthropic`]: Anthropic-compatible API.
//! - [`error`]: OpenAI- and Anthropic-style error envelopes shared by the API handlers.

mod anthropic;
mod error;
mod openai;
mod ops;
mod predictions;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

use crate::state::AppState;

/// The complete route table. State is applied by the caller.
pub(crate) fn router() -> Router<Arc<AppState>> {
    Router::new()
        // Health and management
        .route("/", get(ops::health))
        .route("/health", get(ops::health))
        .route("/status", get(ops::status))
        .route("/drain", post(ops::drain))
        // Muna remote prediction
        .route("/v1/predictions/remote", post(predictions::predictions))
        // OpenAI compatibility
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/images/generations", post(openai::image_generations))
        // Anthropic compatibility
        .route("/v1/messages", post(anthropic::messages))
        // Fallbacks
        .fallback(ops::not_found)
}
