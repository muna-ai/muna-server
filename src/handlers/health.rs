/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use axum::Json;
use serde_json::{json, Value};

/// Liveness probe: a constant answer proving the event loop is turning.
/// Deliberately touches no state so it is free and can never fail.
pub(super) async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
