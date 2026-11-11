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
        .route("/v1/predictions", post(predictions::predictions).layer(limit()))
        // OpenAI compatibility
        .route("/v1/models", get(openai::models))
        .route("/v1/models/{*id}", get(openai::model))
        .route("/v1/chat/completions", post(openai::chat_completions).layer(limit()))
        .route("/v1/embeddings", post(openai::embeddings).layer(limit()))
        .route("/v1/images/generations", post(openai::image_generations).layer(limit()))
        .route("/v1/images/edits", post(openai::image_edits).layer(limit()))
        .route("/v1/audio/transcriptions", post(openai::transcriptions).layer(limit()))
        // Anthropic compatibility
        .route("/v1/messages", post(anthropic::messages).layer(limit()))
        // Fallbacks
        .fallback(not_found::not_found)
        .layer(axum::middleware::from_fn(body::reject_device_headers))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{NodeContext, ServerMode};
    use std::time::Duration;
    use axum::http::StatusCode;

    #[tokio::test]
    async fn native_route_needs_no_assignment_header_and_never_loads_managed_models() {
        let _serial = crate::serving::lease::RUNTIME_TEST_LOCK.lock().unwrap();
        let state = Arc::new(AppState::new(None, ServerMode::Managed(NodeContext {
            node_id: "test-node".into(),
            control_plane_url: "http://127.0.0.1:1".into(),
            heartbeat_interval: Duration::from_secs(5),
            kv_flush_interval: Duration::from_secs(1),
        })).unwrap());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = router().with_state(state.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        let response = client.post(format!("{base}/v1/predictions"))
            .json(&serde_json::json!({"tag":"@test/not-loaded", "inputs":{}}))
            .send().await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(state.registry.snapshot().is_empty());
        for endpoint in ["/v1/predictions", "/v1/chat/completions", "/v1/messages"] {
            let rejected = client.post(format!("{base}{endpoint}"))
                .header("x-muna-device-ids", "0")
                .json(&serde_json::json!({"tag":"@test/not-loaded", "inputs":{}}))
                .send().await.unwrap();
            assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
        }
        let removed = client.post(format!("{base}/v1/predictions/remote"))
            .json(&serde_json::json!({"tag":"@test/not-loaded", "inputs":{}}))
            .send().await.unwrap();
        assert_eq!(removed.status(), StatusCode::NOT_FOUND);
        server.abort();
        let _ = server.await;
        state.lease.shutdown().unwrap();
    }
}
