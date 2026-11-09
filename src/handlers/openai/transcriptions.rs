/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Multipart speech-to-text uploads. Only non-streaming JSON is supported.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, State};
use axum::response::{IntoResponse, Response};
use muna::beta::openai::TranscriptionCreateParams;

use crate::handlers::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{PredictionSample, SampleDetail};
use crate::state::AppState;

pub(crate) async fn transcriptions(
    State(state): State<Arc<AppState>>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let multipart = multipart.map_err(|error| AppError::bad_request(error.to_string()))?;
    let mut params = parse(multipart).await?;
    let model = state.inference_model(&params.model).await?;
    params.devices = model.device_handles()?;
    state.check_in_if_due(&params.model).await;
    state.mark_model_loaded(params.model.clone()).await;
    let admitted = Instant::now();
    let _admission = model.admit().await?;
    let guard = state.dispatcher.acquire(&params.model, &model).await;
    let queue_wait = admitted.elapsed();
    params.acceleration = Some(model.acceleration());
    let muna = model.muna.clone();
    let dispatched = Instant::now();
    let response = predict::run_model(model.clone(), guard, move || async move {
        muna.beta.openai.audio.transcriptions.create(params).await
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

async fn parse(mut multipart: Multipart) -> Result<TranscriptionCreateParams, AppError> {
    let mut params = TranscriptionCreateParams::default();
    let mut seen = HashSet::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::multipart(e.status(), e.body_text()))?
    {
        let name = field.name().unwrap_or_default().to_owned();
        if !seen.insert(name.clone()) {
            return Err(AppError::bad_request(format!(
                "Duplicate transcription field `{name}`."
            )));
        }
        if name == "file" {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| AppError::multipart(e.status(), e.body_text()))?;
            if bytes.is_empty() {
                return Err(AppError::bad_request("Audio file is empty.".into()));
            }
            params.file = bytes.to_vec().into();
            continue;
        }
        if !matches!(
            name.as_str(),
            "model" | "language" | "prompt" | "temperature" | "response_format" | "stream"
        ) {
            return Err(AppError::bad_request(format!(
                "Unsupported transcription field `{name}`."
            )));
        }
        let value = field
            .text()
            .await
            .map_err(|e| AppError::multipart(e.status(), e.body_text()))?;
        match name.as_str() {
            "model" => {
                if value.trim().is_empty() {
                    return Err(AppError::bad_request("The model must not be empty.".into()));
                }
                // Match the control plane's accepted owner/name alias.
                params.model = if !value.starts_with('@') && value.contains('/') {
                    format!("@{value}")
                } else {
                    value
                };
            }
            "language" => params.language = Some(value),
            "prompt" => params.prompt = Some(value),
            "temperature" => {
                let temperature = value.parse::<f64>().map_err(|_| {
                    AppError::bad_request("Temperature must be between 0 and 1.".into())
                })?;
                if !temperature.is_finite() || !(0.0..=1.0).contains(&temperature) {
                    return Err(AppError::bad_request(
                        "Temperature must be between 0 and 1.".into(),
                    ));
                }
                params.temperature = Some(temperature);
            }
            "response_format" if value != "json" => {
                return Err(AppError::bad_request(
                    "Only the JSON transcription response format is supported.".into(),
                ));
            }
            "stream" if value != "false" => {
                return Err(AppError::bad_request(
                    "Only non-streaming transcription is supported (`stream=false`).".into(),
                ));
            }
            _ => {}
        }
    }
    if !seen.contains("file") || !seen.contains("model") {
        return Err(AppError::bad_request(
            "Transcription requires `file` and `model` fields.".into(),
        ));
    }
    Ok(params)
}
