/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{stream, StreamExt};
use muna::beta::openai::{ChatCompletionCreateParams, ChatCompletionMessage};
use muna::types::Acceleration;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::AppState;

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

/// Real chat: run the requested model via muna and relay the result.
pub(crate) async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, AppError> {
    let params = ChatCompletionCreateParams {
        model: req.model,
        messages: req.messages,
        acceleration: Some(Acceleration::LocalGpu),
        max_completion_tokens: req.max_completion_tokens,
        ..Default::default()
    };

    if req.stream {
        stream_chat_completion(state, params).await
    } else {
        create_chat_completion(state, params).await
    }
    .map_err(AppError::from)
}

async fn create_chat_completion(
    state: Arc<AppState>,
    params: ChatCompletionCreateParams,
) -> Result<Response, muna::MunaError> {
    let model = params.model.clone();
    let _guard = state.predict_lock.clone().lock_owned().await;
    let completion = state
        .muna
        .beta
        .openai
        .chat
        .completions
        .create(params)
        .await?;
    mark_model_loaded(&state, model).await;
    Ok(Json(completion).into_response())
}

async fn stream_chat_completion(
    state: Arc<AppState>,
    params: ChatCompletionCreateParams,
) -> Result<Response, muna::MunaError> {
    let model = params.model.clone();
    // Held for the whole Muna stream. Dropping the response body releases the guard.
    let guard = state.predict_lock.clone().lock_owned().await;
    let muna_stream = state
        .muna
        .beta
        .openai
        .chat
        .completions
        .stream(params)
        .await?;
    mark_model_loaded(&state, model).await;
    let event_stream = muna_stream
        .map(move |result| {
            let _guard = &guard;
            match result {
                Ok(chunk) => {
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    Ok::<Event, Infallible>(Event::default().data(json))
                }
                Err(e) => {
                    tracing::warn!("muna stream error: {e}");
                    let json = serde_json::to_string(&muna_error_value(&e)).unwrap_or_default();
                    Ok(Event::default().data(json))
                }
            }
        })
        .chain(stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));

    Ok(Sse::new(event_stream).into_response())
}

async fn mark_model_loaded(state: &AppState, model: String) {
    let mut loaded_models = state.loaded_models.write().await;
    loaded_models.entry(model).or_insert_with(now);
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) struct AppError {
    status: StatusCode,
    body: Value,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

impl From<muna::MunaError> for AppError {
    fn from(e: muna::MunaError) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: muna_error_value(&e),
        }
    }
}

fn muna_error_value(e: &muna::MunaError) -> Value {
    json!({
        "error": {
            "message": e.to_string(),
            "type": "muna_error",
        }
    })
}
