/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::{stream, StreamExt};
use muna::beta::openai::{
    ChatCompletionChunk, ChatCompletionCreateParams, ChatCompletionFunctionTool,
    ChatCompletionMessage, ChatCompletionToolChoice
};
use muna::types::Acceleration;
use serde::Deserialize;

use crate::handlers::error::{muna_error_value, AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail, StreamKind, StreamMeter};
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct ChatCompletionsRequest {
    /// Chat predictor tag.
    model: String,
    /// Messages comprising the conversation so far.
    #[serde(default)]
    messages: Vec<ChatCompletionMessage>,
    /// Whether to stream the response as server-sent events.
    #[serde(default)]
    stream: bool,
    /// Maximum completion tokens. Accepts OpenAI's deprecated
    /// `max_tokens` spelling as an alias.
    #[serde(default, alias = "max_tokens")]
    max_completion_tokens: Option<i32>,
    /// Tools the model may call.
    #[serde(default)]
    tools: Option<Vec<ChatCompletionFunctionTool>>,
    /// Tool choice mode. Unknown modes (`required`, named functions)
    /// fail deserialization and render as 400s.
    #[serde(default)]
    tool_choice: Option<ChatCompletionToolChoice>,
}

/// Chat completions via the muna-rs OpenAI client, wrapped with the model
/// registry (429-on-loading), the sequential dispatch guard, and the
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
    // Time spent acquiring the sequential guard is this surface's
    // admission wait (zero for continuous models).
    let admitted = Instant::now();
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let queue_wait = admitted.elapsed();
    let params = ChatCompletionCreateParams {
        model: req.model,
        messages: req.messages,
        acceleration: Some(Acceleration::LocalGpu),
        max_completion_tokens: req.max_completion_tokens,
        tools: req.tools,
        tool_choice: req.tool_choice,
        ..Default::default()
    };
    let muna = model.muna.clone();
    if req.stream {
        let meter = StreamMeter::new(model.stats.clone(), StreamKind::Llm, queue_wait);
        let rx = predict::stream(move || async move {
            muna.beta.openai.chat.completions.stream(params).await
        });
        // The guard and meter travel with the stream state; dropping the
        // response body (stream end or client disconnect) releases the
        // guard and records the meter's telemetry sample.
        let event_stream = stream::unfold(
            (rx, guard, meter),
            |(mut rx, guard, mut meter)| async move {
                let item = rx.recv().await?;
                let event = match item {
                    Ok(chunk) => {
                        stamp_chunk(&mut meter, &chunk);
                        let json = serde_json::to_string(&chunk).unwrap_or_default();
                        Event::default().data(json)
                    }
                    Err(e) => {
                        tracing::warn!("muna stream error: {e}");
                        let json = serde_json::to_string(&muna_error_value(&e)).unwrap_or_default();
                        Event::default().data(json)
                    }
                };
                Some((Ok::<Event, Infallible>(event), (rx, guard, meter)))
            }
        )
        .chain(stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));
        Ok(Sse::new(event_stream).into_response())
    } else {
        let dispatched = Instant::now();
        let completion = predict::run(move || async move {
            muna.beta.openai.chat.completions.create(params).await
        }).await?;
        drop(guard);
        // Non-streamed chat records `Unary`: whole-response latency has no
        // first-yield boundary and must not fatten the TTFT percentiles.
        model.stats.telemetry.record(PredictionSample {
            at: Instant::now(),
            queue_wait,
            latency: dispatched.elapsed(),
            detail: SampleDetail::Unary,
        });
        Ok(Json(completion).into_response())
    }
}

/// Stamp one chunk on the stream meter: a chunk is content-bearing when
/// any choice's delta carries text (content or reasoning) or tool call
/// fragments, or when it is the terminal usage frame -- role-only
/// wire-consistency frames are not.
fn stamp_chunk(
    meter: &mut StreamMeter,
    chunk: &ChatCompletionChunk
) {
    let has_text = chunk.choices.iter().any(|choice| {
        choice.delta.as_ref().is_some_and(|delta| {
            delta.content.as_deref().is_some_and(|c| !c.is_empty())             ||
            delta.reasoning_content.as_deref().is_some_and(|c| !c.is_empty())   ||
            delta.tool_calls.as_ref().is_some_and(|t| !t.is_empty())
        })
    });
    if let Some(usage) = &chunk.usage {
        meter.on_usage(usage.completion_tokens);
    }
    meter.on_output(has_text || chunk.usage.is_some());
}
