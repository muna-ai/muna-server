/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::AppState;

pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let loaded_models = state.loaded_models.read().await;
    let data: Vec<Value> = loaded_models
        .iter()
        .map(|(model, created)| {
            json!({
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": "muna",
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
    }))
}
