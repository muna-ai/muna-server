/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Model discovery. One route serves `openai.models.list()` and
//! `anthropic.models.list()` alike: each row is the OpenAI `Model` and the
//! Anthropic `ModelInfo` for the same predictor flattened together (the two
//! share no key but `id`), and the envelope carries OpenAI's `object` next
//! to Anthropic's cursor fields. Both objects derive from the predictor
//! signature via `muna::beta`, so the list can never disagree with what the
//! predictor accepts.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path, State};
use muna::beta::{anthropic, openai};
use serde::Serialize;

use crate::handlers::error::{AppError, Json};
use crate::serving::registry::{ModelState, ReadyModel};
use crate::state::{unix_now, AppState};

/// One discovery row: the OpenAI `Model` and the Anthropic `ModelInfo`
/// for the same predictor, flattened into a single JSON object (the field
/// names below never appear in the output).
#[derive(Serialize)]
pub(crate) struct ModelRow {
    #[serde(flatten)]
    openai: openai::Model,
    #[serde(flatten)]
    anthropic: anthropic::ModelInfo,
}

/// The list envelope: OpenAI's `list` object plus Anthropic's cursor page.
/// A node serves few models, so the page is always complete.
#[derive(Serialize)]
pub(crate) struct ModelRows {
    object: &'static str,
    data: Vec<ModelRow>,
    first_id: Option<String>,
    last_id: Option<String>,
    has_more: bool,
}

impl ModelRows {

    fn new(data: Vec<ModelRow>) -> Self {
        Self {
            object: "list",
            first_id: data.first().map(|row| row.openai.id.clone()),
            last_id: data.last().map(|row| row.openai.id.clone()),
            has_more: false,
            data,
        }
    }
}

/// `GET /v1/models`: every model whose engine is warm on this node.
pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Json<ModelRows> {
    let now = unix_now();
    let mut data: Vec<ModelRow> = state.registry.snapshot()
        .into_iter()
        .filter_map(|(tag, model_state)| match model_state {
            ModelState::Ready(model) => Some(row(&tag, &model, now)),
            _ => None,
        })
        .collect();
    // Anthropic lists newest first; OpenAI has no order. Ties (the same
    // load second) fall back to the tag so the order is stable.
    data.sort_by(|a, b| {
        b.openai.created.cmp(&a.openai.created).then_with(|| a.openai.id.cmp(&b.openai.id))
    });
    Json(ModelRows::new(data))
}

/// `GET /v1/models/{*id}`: one model. Tags contain a slash
/// (`@owner/name`), hence the wildcard segment.
pub(crate) async fn model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>
) -> Result<Json<ModelRow>, AppError> {
    let now = unix_now();
    state.registry.snapshot()
        .into_iter()
        .find_map(|(tag, model_state)| match model_state {
            ModelState::Ready(model) if tag == id => Some(row(&tag, &model, now)),
            _ => None,
        })
        .map(Json)
        .ok_or_else(|| AppError::model_not_found(&id))
}

/// A node has no deployment date, so `created` is when the engine finished
/// loading here.
fn row(
    tag: &str,
    model: &ReadyModel,
    now: u64
) -> ModelRow {
    let created = Some(now.saturating_sub(
        Instant::now().duration_since(model.loaded_at).as_secs()
    ));
    ModelRow {
        openai: openai::Model::from_signature(tag, &model.signature, created),
        anthropic: anthropic::ModelInfo::from_signature(tag, &model.signature, created),
    }
}
