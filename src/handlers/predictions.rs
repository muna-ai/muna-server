/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use muna::c;
use muna::types::{
    Acceleration, Dtype, Prediction, RemotePrediction,
    RemoteValue, Tensor, TensorData, Value
};
use muna::MunaError;
use serde::Deserialize;
use serde_json::Value as JsonValue;

use super::error::AppError;
use crate::serving::predict;
use crate::state::AppState;

/// Prediction request.
#[derive(Deserialize)]
pub(crate) struct CreatePredictionRequest {
    /// Predictor tag.
    tag: String,
    /// Prediction inputs keyed by parameter name.
    #[serde(default)]
    inputs: HashMap<String, RemoteValue>,
    /// Requested acceleration.
    #[serde(default)]
    acceleration: Option<Acceleration>,
    /// Whether to stream predictions.
    #[serde(default)]
    stream: bool,
}

/// Create a prediction (routed control-plane traffic and raw clients).
/// Non-streaming requests go through the dispatcher, so buffered models get
/// cross-request batching here.
pub(crate) async fn predictions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreatePredictionRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let acceleration = local_acceleration(req.acceleration.as_ref());
    let tag = req.tag;
    // Parse each remote value input into a native value.
    let mut inputs: HashMap<String, Value> = HashMap::with_capacity(req.inputs.len());
    for (name, remote) in &req.inputs {
        inputs.insert(name.clone(), parse_remote_value(remote)?);
    }
    let model = state.registry.ensure_ready(&tag).await?;
    state.check_in_if_due(&tag).await;
    state.mark_model_loaded(tag.clone()).await;
    if req.stream {
        stream_prediction(state, tag, model, inputs, acceleration).await
    } else {
        let prediction = state.dispatcher
            .create(&tag, &model, inputs, acceleration)
            .await?;
        let remote = to_remote_prediction(prediction)?;
        Ok(Json(remote).into_response())
    }
}

async fn stream_prediction(
    state: Arc<AppState>,
    tag: String,
    model: Arc<crate::serving::registry::ReadyModel>,
    inputs: HashMap<String, Value>,
    acceleration: Acceleration,
) -> Result<Response, AppError> {
    // Streams bypass batching (per-request token streams cannot merge);
    // sequential models still hold their guard for the whole stream.
    let guard = state.dispatcher.acquire(&tag, &model).await;
    let muna = model.muna.clone();
    let stream_tag = tag.clone();
    let rx = predict::stream(move || async move {
        muna.predictions.stream(&stream_tag, inputs, Some(acceleration)).await
    });
    let event_stream = futures_util::stream::unfold(
        (rx, guard, tag),
        |(mut rx, guard, tag)| async move {
            let result = rx.recv().await?;
            let remote = match result {
                Ok(prediction) => to_remote_prediction(prediction)
                    .unwrap_or_else(|e| error_prediction(&tag, &e.to_string())),
                Err(e) => {
                    tracing::warn!("muna prediction stream error: {e}");
                    error_prediction(&tag, &e.to_string())
                }
            };
            let data = serde_json::to_string(&remote).unwrap_or_default();
            let event = Event::default().event("prediction").data(data);
            Some((Ok::<Event, Infallible>(event), (rx, guard, tag)))
        }
    );
    Ok(Sse::new(event_stream).into_response())
}

/// Map a requested acceleration onto a local one:
/// `remote_cpu` runs on the local CPU, everything else on the local GPU.
fn local_acceleration(acceleration: Option<&Acceleration>) -> Acceleration {
    match acceleration {
        Some(Acceleration::RemoteCpu) => Acceleration::LocalCpu,
        Some(Acceleration::Adaptive(s)) if s == "remote_cpu" => Acceleration::LocalCpu,
        _ => Acceleration::LocalGpu,
    }
}

fn to_remote_prediction(prediction: Prediction) -> Result<RemotePrediction, MunaError> {
    let results = match prediction.results {
        Some(values) => {
            let mut remote_values = Vec::with_capacity(values.len());
            for value in &values {
                remote_values.push(create_remote_value(value)?);
            }
            Some(remote_values)
        }
        None => None,
    };
    Ok(RemotePrediction {
        id: prediction.id,
        tag: prediction.tag,
        created: prediction.created,
        results,
        latency: prediction.latency,
        error: prediction.error,
        logs: prediction.logs,
    })
}

fn error_prediction(tag: &str, message: &str) -> RemotePrediction {
    RemotePrediction {
        id: create_prediction_id(),
        tag: tag.to_string(),
        created: crate::state::unix_now().to_string(),
        results: None,
        latency: None,
        error: Some(message.to_string()),
        logs: None,
    }
}

fn parse_remote_value(remote: &RemoteValue) -> Result<Value, MunaError> {
    if remote.dtype == Dtype::Null {
        return Ok(Value::Null);
    }
    let url = remote
        .data
        .as_deref()
        .ok_or_else(|| MunaError::Prediction("Remote value has no data URL".into()))?;
    let buffer = decode_data_url(url)?;
    match remote.dtype {
        Dtype::Null => Ok(Value::Null),
        dtype if is_tensor_dtype(dtype) => {
            let fxn_value = c::Value::from_bytes(&buffer, "application/vnd.muna.tensor")?;
            fxn_value.to_object()
        }
        Dtype::String => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            Ok(Value::String(s))
        }
        Dtype::List => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            let v: Vec<JsonValue> = serde_json::from_str(&s)?;
            Ok(Value::List(v))
        }
        Dtype::Dict => {
            let s = String::from_utf8(buffer)
                .map_err(|e| MunaError::Prediction(format!("UTF-8 decode error: {e}")))?;
            let v: serde_json::Map<String, JsonValue> = serde_json::from_str(&s)?;
            Ok(Value::Dict(v))
        }
        Dtype::Image => {
            let fxn_value = c::Value::from_bytes(&buffer, "image/*")?;
            fxn_value.to_object()
        }
        Dtype::ArrayList => {
            let fxn_value = c::Value::from_bytes(&buffer, "application/x-npz")?;
            fxn_value.to_object()
        }
        Dtype::ImageList => {
            let fxn_value = c::Value::from_bytes(&buffer, "image/avif")?;
            fxn_value.to_object()
        }
        Dtype::Binary => Ok(Value::Binary(buffer)),
        dtype => Err(MunaError::Prediction(format!(
            "Cannot deserialize remote value with type `{dtype:?}`"
        ))),
    }
}

fn create_remote_value(value: &Value) -> Result<RemoteValue, MunaError> {
    match value {
        Value::Null => Ok(RemoteValue {
            data: None,
            dtype: Dtype::Null,
        }),
        Value::Float(v) => create_remote_value(&Value::Tensor(Tensor {
            data: TensorData::Float32(vec![*v]),
            shape: vec![],
        })),
        Value::Double(v) => create_remote_value(&Value::Tensor(Tensor {
            data: TensorData::Float32(vec![*v as f32]),
            shape: vec![],
        })),
        Value::Int(v) => create_remote_value(&Value::Tensor(Tensor {
            data: TensorData::Int32(vec![*v]),
            shape: vec![],
        })),
        Value::Long(v) => create_remote_value(&Value::Tensor(Tensor {
            data: TensorData::Int64(vec![*v]),
            shape: vec![],
        })),
        Value::Bool(v) => create_remote_value(&Value::Tensor(Tensor {
            data: TensorData::Bool(vec![*v]),
            shape: vec![],
        })),
        Value::Tensor(tensor) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "application/octet-stream")),
                dtype: tensor.data.dtype(),
            })
        }
        Value::String(s) => Ok(RemoteValue {
            data: Some(encode_data_url(s.as_bytes(), "text/plain")),
            dtype: Dtype::String,
        }),
        Value::List(v) => {
            let json = serde_json::to_string(v)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(json.as_bytes(), "application/json")),
                dtype: Dtype::List,
            })
        }
        Value::Dict(v) => {
            let json = serde_json::to_string(v)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(json.as_bytes(), "application/json")),
                dtype: Dtype::Dict,
            })
        }
        Value::Image(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "image/png")),
                dtype: Dtype::Image,
            })
        }
        Value::ArrayList(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "application/x-npz")),
                dtype: Dtype::ArrayList,
            })
        }
        Value::ImageList(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "image/avif")),
                dtype: Dtype::ImageList,
            })
        }
        Value::Binary(bytes) => Ok(RemoteValue {
            data: Some(encode_data_url(bytes, "application/octet-stream")),
            dtype: Dtype::Binary,
        }),
    }
}

fn encode_data_url(buffer: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", BASE64.encode(buffer))
}

fn decode_data_url(url: &str) -> Result<Vec<u8>, MunaError> {
    if let Some(rest) = url.strip_prefix("data:") {
        if let Some((_mime, encoded)) = rest.split_once(";base64,") {
            return BASE64
                .decode(encoded)
                .map_err(|e| MunaError::Prediction(format!("Base64 decode error: {e}")));
        }
    }
    Err(MunaError::Prediction(
        "Unsupported value data URL; only inline base64 `data:` URLs are supported".into(),
    ))
}

/// Native dtypes that round-trip through the fxnc tensor serializer.
fn is_tensor_dtype(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::BFloat16
            | Dtype::Float16
            | Dtype::Float32
            | Dtype::Float64
            | Dtype::Int8
            | Dtype::Int16
            | Dtype::Int32
            | Dtype::Int64
            | Dtype::Uint8
            | Dtype::Uint16
            | Dtype::Uint32
            | Dtype::Uint64
            | Dtype::Complex64
            | Dtype::Complex128
            | Dtype::Bool
    )
}

fn create_prediction_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("pred_{nanos}")
}
