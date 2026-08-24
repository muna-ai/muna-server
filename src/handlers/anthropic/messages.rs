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
use futures_util::stream;
use muna::beta::anthropic::{MessageContent, MessageCreateParams, MessageParam};
use muna::types::Acceleration;
use serde::Deserialize;

use crate::handlers::error::{anthropic_error_value, AnthropicError, AppError};
use crate::serving::predict;
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct MessagesRequest {
    model: String,
    max_tokens: i32,
    #[serde(default)]
    messages: Vec<MessageParam>,
    #[serde(default)]
    system: Option<MessageContent>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i32>,
    #[serde(default)]
    top_p: Option<f32>,
}

/// Messages via the muna-rs Anthropic client, wrapped with the model
/// registry (429-on-loading), the sequential dispatch guard, and the
/// blocking prediction executor.
pub(crate) async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<Response, AnthropicError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30).into());
    }
    let model = state.registry.ensure_ready(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let params = MessageCreateParams {
        model: req.model,
        max_tokens: req.max_tokens,
        messages: req.messages,
        system: req.system,
        stop_sequences: req.stop_sequences,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        acceleration: Some(Acceleration::LocalGpu),
    };
    let muna = model.muna.clone();
    if req.stream {
        let rx = predict::stream(move || async move {
            muna.beta.anthropic.messages.stream(params).await
        });
        // Anthropic SSE frames are named events with no `[DONE]` terminator:
        // the stream simply ends after `message_stop`. Mid-stream errors are
        // emitted as `event: error` with the Anthropic error envelope. The
        // guard travels with the stream state; dropping the response body
        // (client disconnect) releases it.
        let event_stream = stream::unfold((rx, guard), |(mut rx, guard)| async move {
            let item = rx.recv().await?;
            let event = match item {
                Ok(message_event) => {
                    let json = serde_json::to_string(&message_event).unwrap_or_default();
                    Event::default().event(message_event.event_type()).data(json)
                }
                Err(e) => {
                    tracing::warn!("muna stream error: {e}");
                    let json =
                        serde_json::to_string(&anthropic_error_value(&e)).unwrap_or_default();
                    Event::default().event("error").data(json)
                }
            };
            Some((Ok::<Event, Infallible>(event), (rx, guard)))
        });
        Ok(Sse::new(event_stream).into_response())
    } else {
        let message = predict::run(move || async move {
            muna.beta.anthropic.messages.create(params).await
        }).await?;
        drop(guard);
        Ok(Json(message).into_response())
    }
}
