/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{stream, StreamExt};
use muna::beta::openai::{ChatCompletionCreateParams, ChatCompletionMessage};
use muna::types::Acceleration;
use serde::Deserialize;

use super::error::{muna_error_value, AppError};
use crate::serving::predict;
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct ChatCompletionsRequest {
    model: String,
    #[serde(default)]
    messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_tokens")]
    max_completion_tokens: Option<i32>,
}

/// Chat completions via the muna-rs OpenAI client, wrapped with the model
/// registry (503-on-loading), the sequential dispatch guard, and the
/// blocking prediction executor.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let model = state.registry.ensure_ready(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let params = ChatCompletionCreateParams {
        model: req.model,
        messages: req.messages,
        acceleration: Some(Acceleration::LocalGpu),
        max_completion_tokens: req.max_completion_tokens,
        ..Default::default()
    };
    let muna = state.muna.clone();
    if req.stream {
        let rx = predict::stream(move || async move {
            muna.beta.openai.chat.completions.stream(params).await
        });
        // The guard travels with the stream state; dropping the response
        // body (client disconnect) releases it.
        let event_stream = stream::unfold((rx, guard), |(mut rx, guard)| async move {
            let item = rx.recv().await?;
            let event = match item {
                Ok(chunk) => {
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    Event::default().data(json)
                }
                Err(e) => {
                    tracing::warn!("muna stream error: {e}");
                    let json = serde_json::to_string(&muna_error_value(&e)).unwrap_or_default();
                    Event::default().data(json)
                }
            };
            Some((Ok::<Event, Infallible>(event), (rx, guard)))
        })
        .chain(stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));
        Ok(Sse::new(event_stream).into_response())
    } else {
        let completion = predict::run(move || async move {
            muna.beta.openai.chat.completions.create(params).await
        }).await?;
        drop(guard);
        Ok(Json(completion).into_response())
    }
}
