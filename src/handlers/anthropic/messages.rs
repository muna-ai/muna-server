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
use axum::Json;
use futures_util::stream;
use muna::beta::anthropic::{
    MessageContent, MessageCreateParams, MessageParam,
    RawMessageStreamEvent
};
use muna::types::Acceleration;
use serde::Deserialize;

use crate::handlers::error::{
    anthropic_error_value, AnthropicError,
    AnthropicJson, AppError
};
use crate::serving::predict;
use crate::serving::stats::{
    PredictionSample, SampleDetail, StreamKind,
    StreamMeter
};
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
    AnthropicJson(req): AnthropicJson<MessagesRequest>,
) -> Result<Response, AnthropicError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30).into());
    }
    let model = state.registry.ensure_ready(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    // Time spent acquiring the sequential guard is this surface's
    // admission wait (zero for continuous models).
    let admitted = Instant::now();
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let queue_wait = admitted.elapsed();
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
        let meter = StreamMeter::new(model.stats.clone(), StreamKind::Llm, queue_wait);
        let rx = predict::stream(move || async move {
            muna.beta.anthropic.messages.stream(params).await
        });
        // Anthropic SSE frames are named events with no `[DONE]` terminator:
        // the stream simply ends after `message_stop`. Mid-stream errors are
        // emitted as `event: error` with the Anthropic error envelope. The
        // guard and meter travel with the stream state; dropping the
        // response body (stream end or client disconnect) releases the
        // guard and records the meter's telemetry sample.
        let event_stream = stream::unfold(
            (rx, guard, meter),
            |(mut rx, guard, mut meter)| async move {
                let item = rx.recv().await?;
                let event = match item {
                    Ok(message_event) => {
                        stamp_event(&mut meter, &message_event);
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
                Some((Ok::<Event, Infallible>(event), (rx, guard, meter)))
            }
        );
        Ok(Sse::new(event_stream).into_response())
    } else {
        let dispatched = Instant::now();
        let message = predict::run(move || async move {
            muna.beta.anthropic.messages.create(params).await
        }).await?;
        drop(guard);
        // Non-streamed messages record `Unary`: whole-response latency has
        // no first-yield boundary and must not fatten the TTFT percentiles.
        model.stats.telemetry.record(PredictionSample {
            at: Instant::now(),
            queue_wait,
            latency: dispatched.elapsed(),
            detail: SampleDetail::Unary,
        });
        Ok(Json(message).into_response())
    }
}

/// Stamp one stream event on the meter: `content_block_delta` frames are
/// content-bearing; `message_delta` carries the cumulative output token
/// count used for the yield-invariant interval normalization. Start/stop
/// envelope frames are wire consistency only.
fn stamp_event(
    meter: &mut StreamMeter,
    event: &RawMessageStreamEvent
) {
    match event {
        RawMessageStreamEvent::ContentBlockDelta { .. } => meter.on_output(true),
        RawMessageStreamEvent::MessageDelta { usage, .. } => {
            meter.on_usage(usage.output_tokens);
            meter.on_output(true);
        }
        _ => meter.on_output(false),
    }
}
