/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use muna::beta::adapters::{Adapter, EncodingFormat, OpenAIEmbeddingsOptions};
use serde::Deserialize;

use crate::handlers::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail};
use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct EmbeddingsRequest {
    #[serde(flatten)]
    _device_selection: crate::handlers::body::DeviceSelection,
    input: EmbeddingsInput,
    model: String,
    dimensions: Option<i32>,
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingsInput {
    String(String),
    Strings(Vec<String>),
}

impl EmbeddingsInput {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::String(input) => vec![input],
            Self::Strings(input) => input,
        }
    }
}

pub(crate) async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let input = req.input.into_vec();
    let encoding_format = parse_encoding_format(req.encoding_format.as_deref())?;
    let model = state.inference_model(&req.model).await?;
    state.check_in_if_due(&req.model).await;
    state.mark_model_loaded(req.model.clone()).await;
    // Time spent acquiring the sequential guard is this surface's
    // admission wait (zero for continuous models).
    let admitted = Instant::now();
    let _admission = model.admit().await?;
    let guard = state.dispatcher.acquire(&req.model, &model).await;
    let queue_wait = admitted.elapsed();
    let adapter = model.adapters.embeddings()?;
    let inputs = adapter.inputs(input, req.dimensions);
    let muna = model.muna.clone();
    let tag = req.model;
    let devices = model.device_handles()?;
    let acceleration = model.acceleration();
    let dispatched = Instant::now();
    let response = predict::run_model(model.clone(), guard, move || async move {
        let prediction = muna
            .predictions
            .create(
                &tag,
                Some(inputs),
                Some(acceleration),
                devices.as_deref(),
                None,
                None,
                None,
                Some(true),
            )
            .await?;
        adapter
            .adapt(
                prediction,
                Some(OpenAIEmbeddingsOptions {
                    encoding_format: encoding_format.unwrap_or_default(),
                }),
            )
            .await
    })
    .await?;
    model.stats.telemetry.record(PredictionSample {
        at: Instant::now(),
        queue_wait,
        latency: dispatched.elapsed(),
        detail: SampleDetail::Unary,
    });
    Ok(Json(response).into_response())
}

fn parse_encoding_format(value: Option<&str>) -> Result<Option<EncodingFormat>, AppError> {
    match value {
        Some("float") => Ok(Some(EncodingFormat::Float)),
        Some("base64") => Ok(Some(EncodingFormat::Base64)),
        Some(value) => Err(AppError::bad_request(format!(
            "unsupported encoding_format `{value}`"
        ))),
        None => Ok(None),
    }
}
