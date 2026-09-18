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
use muna::beta::openai::{ChatCompletionChunk, ChatCompletionCreateParams};
use muna::types::Acceleration;

use crate::handlers::body::ChatBody;
use crate::handlers::error::{muna_error_value, AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail, StreamKind, StreamMeter};
use crate::state::AppState;

/// Chat completions via the muna-rs OpenAI client, wrapped with the model
/// registry (429-on-loading), the sequential dispatch guard, and the
/// blocking prediction executor.
///
/// The body deserializes straight into muna-rs's request params (see
/// `ChatBody`); unknown `tool_choice` modes (`required`, named functions)
/// fail deserialization there and render as 400s.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatBody<ChatCompletionCreateParams>>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let ChatBody { stream: streaming, mut params } = body;
    params.acceleration = Some(Acceleration::LocalGpu);
    let model = state.registry.ensure_ready(&params.model).await?;
    state.check_in_if_due(&params.model).await;
    state.mark_model_loaded(params.model.clone()).await;
    // Time spent acquiring the sequential guard is this surface's
    // admission wait (zero for continuous models).
    let admitted = Instant::now();
    let guard = state.dispatcher.acquire(&params.model, &model).await;
    let queue_wait = admitted.elapsed();
    let muna = model.muna.clone();
    if streaming {
        let meter = StreamMeter::new(
            model.stats.clone(),
            StreamKind::Llm,
            queue_wait
        );
        // The guard rides with the pump onto the blocking thread and is
        // released once the native stream is dropped; the meter travels
        // with the stream state and records its telemetry sample when the
        // response body drops (stream end or client disconnect).
        let mut rx = predict::stream(guard, move || async move {
            muna.beta.openai.chat.completions.stream(params).await
        });
        // Commit the response status on the first item. Failures raised
        // before the predictor's first output (context length, malformed
        // messages) arrive here as `Err` and become a proper 400 / 500 body
        // instead of a 200 stream carrying an error frame. Errors after
        // this point are mid-stream and stay in-band, as OpenAI's do.
        let first = match rx.recv().await {
            Some(Ok(chunk)) => chunk,
            Some(Err(e)) => return Err(e.into()),
            None => return Err(AppError::internal(
                "prediction stream ended before producing output".into()
            )),
        };
        let event_stream = stream::unfold(
            (rx, meter, Some(first)),
            |(mut rx, mut meter, mut pending)| async move {
                let item = match pending.take() {
                    Some(chunk) => Ok(chunk),
                    None => rx.recv().await?,
                };
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
                Some((Ok::<Event, Infallible>(event), (rx, meter, pending)))
            }
        )
        .chain(stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));
        Ok(Sse::new(event_stream).into_response())
    } else {
        let dispatched = Instant::now();
        let completion = predict::run(guard, move || async move {
            muna.beta.openai.chat.completions.create(params).await
        }).await?;
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
