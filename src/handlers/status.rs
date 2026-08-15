/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use axum::extract::State;
use axum::Json;

use crate::control::protocol::NodeStatus;
use crate::state::AppState;

/// Full node status: per-model state, GPU metrics, uptime. The same payload
/// the heartbeat POSTs to the control plane.
pub(crate) async fn status(State(state): State<Arc<AppState>>) -> Json<NodeStatus> {
    Json(NodeStatus::collect(&state))
}
