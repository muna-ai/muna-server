/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Operational endpoints consumed by supervisors and the control plane,
//! not by API users: liveness, node status, drain, and the fallback route.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};

use crate::control::protocol::NodeStatus;
use crate::state::AppState;

/// Liveness probe: a constant answer proving the event loop is turning.
/// Deliberately touches no state so it is free and can never fail.
pub(super) async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

/// Full node status: per-model state, GPU metrics, uptime. The same payload
/// the heartbeat POSTs to the control plane.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Json<NodeStatus> {
    Json(NodeStatus::collect(&state))
}

/// Stop accepting new inference requests; in-flight work drains naturally.
pub(super) async fn drain(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.set_draining(true);
    tracing::info!("draining: new inference requests will be rejected");
    Json(json!({ "status": "draining" }))
}

pub(super) async fn not_found() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": "unknown route (muna-server)",
                "type": "not_found",
            }
        })),
    )
}
