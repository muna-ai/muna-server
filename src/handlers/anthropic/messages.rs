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
use muna::beta::anthropic::{MessageCreateParams, RawMessageStreamEvent};
use muna::types::Acceleration;

use crate::handlers::body::ChatBody;
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

/// Messages via the muna-rs Anthropic client, wrapped with the model
/// registry (429-on-loading), the sequential dispatch guard, and the
/// blocking prediction executor.
///
/// The body deserializes straight into muna-rs's request params (see
/// `ChatBody`); `max_tokens` is required there, as on the Anthropic API,
/// so a body without it renders as a 400.
pub(crate) async fn messages(
    State(state): State<Arc<AppState>>,
    AnthropicJson(body): AnthropicJson<ChatBody<MessageCreateParams>>,
) -> Result<Response, AnthropicError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30).into());
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
        // released once the native stream is dropped.
        let mut rx = predict::stream(guard, move || async move {
            muna.beta.anthropic.messages.stream(params).await
        });
        // Commit the response status on the first item. Failures raised
        // before the predictor's first output (context length, malformed
        // messages) arrive here as `Err` and become a proper 400 / 500 body
        // instead of a 200 stream carrying an error frame. Errors after
        // this point are mid-stream and stay in-band, as Anthropic's do.
        let first = match rx.recv().await {
            Some(Ok(message_event)) => message_event,
            Some(Err(e)) => return Err(AppError::from(e).into()),
            None => return Err(AppError::internal(
                "prediction stream ended before producing output".into()
            ).into()),
        };
        // Anthropic SSE frames are named events with no `[DONE]` terminator:
        // the stream simply ends after `message_stop`. Mid-stream errors are
        // emitted as `event: error` with the Anthropic error envelope. The
        // meter travels with the stream state and records its telemetry
        // sample when the response body drops (stream end or client
        // disconnect).
        let event_stream = stream::unfold(
            (rx, meter, Some(first)),
            |(mut rx, mut meter, mut pending)| async move {
                let item = match pending.take() {
                    Some(message_event) => Ok(message_event),
                    None => rx.recv().await?,
                };
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
                Some((Ok::<Event, Infallible>(event), (rx, meter, pending)))
            }
        );
        Ok(Sse::new(event_stream).into_response())
    } else {
        let dispatched = Instant::now();
        let message = predict::run(guard, move || async move {
            muna.beta.anthropic.messages.create(params).await
        }).await?;
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
