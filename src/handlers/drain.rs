/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::state::AppState;

/// Stop accepting new inference requests; in-flight work drains naturally.
pub(crate) async fn drain(State(state): State<Arc<AppState>>) -> Json<Value> {
    state.set_draining(true);
    tracing::info!("draining: new inference requests will be rejected");
    Json(json!({ "status": "draining" }))
}
