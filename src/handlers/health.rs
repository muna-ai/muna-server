/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use axum::Json;
use serde_json::{json, Value};

pub(crate) async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}
