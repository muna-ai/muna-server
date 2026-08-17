/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::serving::registry::ModelState;
use crate::state::{unix_now, AppState};

pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let now = unix_now();
    let data: Vec<Value> = state.registry.snapshot()
        .into_iter()
        .filter_map(|(tag, model_state)| match model_state {
            ModelState::Ready(model) => {
                let created = now.saturating_sub(
                    Instant::now().duration_since(model.loaded_at).as_secs()
                );
                Some(json!({
                    "id": tag,
                    "object": "model",
                    "created": created,
                    "owned_by": "muna",
                }))
            }
            _ => None,
        })
        .collect();
    Json(json!({
        "object": "list",
        "data": data,
    }))
}
