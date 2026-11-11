/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use muna::c;
use muna::types::{
    Acceleration, Dtype, Prediction, RemoteData, RemotePrediction,
    RemoteValue, Tensor, TensorData, Value,
};
use muna::MunaError;
use serde::Deserialize;
use serde_json::Value as JsonValue;

use super::error::{AppError, Json};
use crate::serving::predict;
use crate::serving::stats::{StreamKind, StreamMeter};
use crate::state::AppState;

/// Response serialization only; never part of predictor arguments or shared state.
#[derive(Clone, Default)]
struct ImageEncoding {
    format: Option<String>,
    quality: Option<u8>,
}

impl ImageEncoding {
    fn from_headers(headers: &HeaderMap) -> Result<Self, MunaError> {
        let get = |name: &str| -> Result<Option<&str>, MunaError> {
            headers
                .get(name)
                .map(|value| {
                    value
                        .to_str()
                        .map_err(|_| MunaError::InvalidInput(format!("Invalid {name} header.")))
                })
                .transpose()
        };
        let format = get("x-muna-image-output-format")?.map(str::to_owned);
        if format
            .as_deref()
            .is_some_and(|format| !matches!(format, "png" | "jpeg" | "webp"))
        {
            return Err(MunaError::InvalidInput(
                "Image output format must be png, jpeg, or webp.".into(),
            ));
        }
        let quality = get("x-muna-image-output-quality")?
            .map(|value| {
                value
                    .parse::<u8>()
                    .ok()
                    .filter(|value| *value <= 100)
                    .ok_or_else(|| {
                        MunaError::InvalidInput(
                            "Image output quality must be between 0 and 100.".into(),
                        )
                    })
            })
            .transpose()?;
        if quality.is_some() && format.is_none() {
            return Err(MunaError::InvalidInput(
                "Image output quality requires an image output format.".into(),
            ));
        }
        Ok(Self { format, quality })
    }

    fn encode(&self, image: &muna::types::Image) -> Result<RemoteValue, MunaError> {
        let format = self.format.as_deref().unwrap_or("png");
        let mut mime = format!("image/{format}");
        if let Some(quality) = self.quality {
            mime.push_str(&format!(";q={:.2}", quality as f32 / 100.0));
        }
        let buffer = c::Value::from_object(&Value::Image(image.clone()))?.serialize(Some(&mime))?;
        Ok(RemoteValue {
            dtype: Dtype::Image,
            data: Some(encode_data_url(&buffer, &format!("image/{format}")).into()),
        })
    }
}

/// Prediction request.
#[derive(Deserialize)]
pub(crate) struct CreatePredictionRequest {
    #[serde(flatten)]
    _device_selection: crate::handlers::body::DeviceSelection,
    /// Predictor tag.
    tag: String,
    /// Prediction inputs keyed by parameter name.
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
    headers: HeaderMap,
    Json(req): Json<CreatePredictionRequest>,
) -> Result<Response, AppError> {
    if state.is_draining() {
        return Err(AppError::unavailable("node is draining".into(), 30));
    }
    let encoding = ImageEncoding::from_headers(&headers)?;
    let tag = req.tag;
    // Parse each remote value input into a native value.
    let remote_inputs = req.inputs;
    let inputs = tokio::task::spawn_blocking(move || {
        remote_inputs
            .into_iter()
            .map(|(name, remote)| Ok((name, parse_remote_value(&remote)?)))
            .collect::<Result<HashMap<String, Value>, MunaError>>()
    })
    .await
    .map_err(|error| AppError::internal(error.to_string()))??;
    let model = state.inference_model(&tag).await?;
    let acceleration = if model.devices.as_ref().is_some_and(Vec::is_empty) {
        model.acceleration()
    } else {
        local_acceleration(req.acceleration.as_ref())
    };
    state.check_in_if_due(&tag).await;
    state.mark_model_loaded(tag.clone()).await;
    if req.stream {
        stream_prediction(state, tag, model, inputs, acceleration, encoding).await
    } else {
        let prediction = state
            .dispatcher
            .create(&tag, &model, inputs, acceleration)
            .await?;
        let remote =
            tokio::task::spawn_blocking(move || to_remote_prediction(prediction, &encoding))
                .await
                .map_err(|error| AppError::internal(error.to_string()))??;
        Ok(Json(remote).into_response())
    }
}

async fn stream_prediction(
    state: Arc<AppState>,
    tag: String,
    model: Arc<crate::serving::registry::ReadyModel>,
    inputs: HashMap<String, Value>,
    acceleration: Acceleration,
    encoding: ImageEncoding,
) -> Result<Response, AppError> {
    // Streams bypass batching (per-request token streams cannot merge);
    // sequential models still hold their guard for the whole stream: it
    // rides with the pump onto the blocking thread and is released once
    // the native stream is dropped. Time spent acquiring the guard is this
    // surface's admission wait.
    let admitted = std::time::Instant::now();
    let _admission = model.admit().await?;
    let guard = state.dispatcher.acquire(&tag, &model).await;
    // Raw frames are opaque, so only time to the first frame is kept; the
    // meter records its sample when the stream state drops.
    let meter = StreamMeter::new(model.stats.clone(), StreamKind::Raw, admitted.elapsed());
    let muna = model.muna.clone();
    let stream_tag = tag.clone();
    let devices = model.device_handles()?;
    let rx = predict::stream_model(model.clone(), guard, move || async move {
        muna.predictions
            .stream(&stream_tag, inputs, Some(acceleration), devices.as_deref())
            .await
    });
    let event_stream = futures_util::stream::unfold(
        (rx, tag, meter, encoding),
        |(mut rx, tag, mut meter, encoding)| async move {
            let result = rx.recv().await?;
            let remote = match result {
                Ok(prediction) => {
                    meter.on_output(true);
                    to_remote_prediction(prediction, &encoding)
                        .unwrap_or_else(|e| error_prediction(&tag, &e.to_string()))
                }
                Err(e) => {
                    tracing::warn!("muna prediction stream error: {e}");
                    error_prediction(&tag, &e.to_string())
                }
            };
            let data = serde_json::to_string(&remote).unwrap_or_default();
            let event = Event::default().event("prediction").data(data);
            Some((Ok::<Event, Infallible>(event), (rx, tag, meter, encoding)))
        },
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

fn to_remote_prediction(
    prediction: Prediction,
    encoding: &ImageEncoding,
) -> Result<RemotePrediction, MunaError> {
    let results = match prediction.results {
        Some(values) => {
            let mut remote_values = Vec::with_capacity(values.len());
            for value in &values {
                remote_values.push(create_remote_value(value, encoding)?);
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
    if let Some(RemoteData::Sequence(items)) = &remote.data {
        if remote.dtype != Dtype::ImageList || items.len() > 1024 {
            return Err(MunaError::InvalidInput(
                "Only image_list supports composed data (at most 1024 images).".into(),
            ));
        }
        let mut images = Vec::with_capacity(items.len());
        for item in items {
            let url = item
                .data
                .as_ref()
                .and_then(RemoteData::as_str)
                .filter(|_| item.dtype == Dtype::Image)
                .ok_or_else(|| {
                    MunaError::InvalidInput("Image lists require encoded image children.".into())
                })?;
            let buffer = decode_data_url(url)?;
            match c::Value::from_bytes(&buffer, "image/*")
                .and_then(|value| value.to_object())
                .map_err(|error| MunaError::InvalidInput(format!("Cannot decode image: {error}")))?
            {
                Value::Image(image) => images.push(image),
                _ => {
                    return Err(MunaError::InvalidInput(
                        "Child did not decode to an image.".into(),
                    ))
                }
            }
        }
        return Ok(Value::ImageList(images));
    }

    if remote.dtype == Dtype::Null {
        return Ok(Value::Null);
    }
    let url = remote
        .data
        .as_ref()
        .and_then(RemoteData::as_str)
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

fn create_remote_value(value: &Value, encoding: &ImageEncoding) -> Result<RemoteValue, MunaError> {
    match value {
        Value::Null => Ok(RemoteValue {
            data: None,
            dtype: Dtype::Null,
        }),
        Value::Float(v) => create_remote_value(
            &Value::Tensor(Tensor {
                data: TensorData::Float32(vec![*v]),
                shape: vec![],
            }),
            encoding,
        ),
        Value::Double(v) => create_remote_value(
            &Value::Tensor(Tensor {
                data: TensorData::Float32(vec![*v as f32]),
                shape: vec![],
            }),
            encoding,
        ),
        Value::Int(v) => create_remote_value(
            &Value::Tensor(Tensor {
                data: TensorData::Int32(vec![*v]),
                shape: vec![],
            }),
            encoding,
        ),
        Value::Long(v) => create_remote_value(
            &Value::Tensor(Tensor {
                data: TensorData::Int64(vec![*v]),
                shape: vec![],
            }),
            encoding,
        ),
        Value::Bool(v) => create_remote_value(
            &Value::Tensor(Tensor {
                data: TensorData::Bool(vec![*v]),
                shape: vec![],
            }),
            encoding,
        ),
        Value::Tensor(tensor) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "application/octet-stream").into()),
                dtype: tensor.data.dtype(),
            })
        }
        Value::String(s) => Ok(RemoteValue {
            data: Some(encode_data_url(s.as_bytes(), "text/plain").into()),
            dtype: Dtype::String,
        }),
        Value::List(v) => {
            let json = serde_json::to_string(v)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(json.as_bytes(), "application/json").into()),
                dtype: Dtype::List,
            })
        }
        Value::Dict(v) => {
            let json = serde_json::to_string(v)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(json.as_bytes(), "application/json").into()),
                dtype: Dtype::Dict,
            })
        }
        Value::Image(image) if encoding.format.is_some() => encoding.encode(image),
        Value::Image(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "image/png").into()),
                dtype: Dtype::Image,
            })
        }
        Value::ArrayList(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "application/x-npz").into()),
                dtype: Dtype::ArrayList,
            })
        }
        Value::ImageList(images) if encoding.format.is_some() => Ok(RemoteValue {
            dtype: Dtype::ImageList,
            data: Some(RemoteData::Sequence(
                images
                    .iter()
                    .map(|image| encoding.encode(image))
                    .collect::<Result<_, _>>()?,
            )),
        }),
        Value::ImageList(_) => {
            let buffer = c::Value::from_object(value)?.serialize(None)?;
            Ok(RemoteValue {
                data: Some(encode_data_url(&buffer, "image/avif").into()),
                dtype: Dtype::ImageList,
            })
        }
        Value::Binary(bytes) => Ok(RemoteValue {
            data: Some(encode_data_url(bytes, "application/octet-stream").into()),
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

#[cfg(test)]
mod image_transport_tests {
    use super::*;

    fn image() -> muna::types::Image {
        muna::types::Image {
            width: 32,
            height: 32,
            channels: 3,
            data: (0..32 * 32 * 3)
                .map(|i| ((i * 71 + i / 17) % 256) as u8)
                .collect(),
        }
    }

    fn encoding(format: &str, quality: &str) -> ImageEncoding {
        let mut headers = HeaderMap::new();
        headers.insert("x-muna-image-output-format", format.parse().unwrap());
        headers.insert("x-muna-image-output-quality", quality.parse().unwrap());
        ImageEncoding::from_headers(&headers).unwrap()
    }

    #[test]
    fn composed_outputs_decode_in_order_and_preserve_legacy_avif() {
        let original = image();
        let value = Value::ImageList(vec![original.clone(), original.clone()]);
        let png = create_remote_value(&value, &encoding("png", "80")).unwrap();
        let Some(RemoteData::Sequence(items)) = &png.data else {
            panic!("expected sequence")
        };
        assert_eq!(items.len(), 2);
        match parse_remote_value(&png).unwrap() {
            Value::ImageList(images) => {
                assert_eq!(images, vec![original.clone(), original.clone()])
            }
            _ => panic!("expected images"),
        }
        let legacy = create_remote_value(&value, &ImageEncoding::default()).unwrap();
        assert!(legacy
            .data
            .as_ref()
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("data:image/avif;"));
        assert!(
            matches!(parse_remote_value(&legacy).unwrap(), Value::ImageList(images) if images.len() == 2)
        );
        for format in ["jpeg", "webp"] {
            let remote = create_remote_value(&value, &encoding(format, "80")).unwrap();
            assert!(
                matches!(parse_remote_value(&remote).unwrap(), Value::ImageList(images) if images.len() == 2)
            );
        }
    }

    #[test]
    fn quality_reaches_the_existing_native_encoder() {
        let value = Value::Image(image());
        let low = create_remote_value(&value, &encoding("jpeg", "10")).unwrap();
        let high = create_remote_value(&value, &encoding("jpeg", "100")).unwrap();
        let low = decode_data_url(low.data.as_ref().unwrap().as_str().unwrap()).unwrap();
        let high = decode_data_url(high.data.as_ref().unwrap().as_str().unwrap()).unwrap();
        assert!(high.len() > low.len());
    }

    #[test]
    fn rejects_invalid_headers_and_nested_children() {
        for (name, value) in [
            ("x-muna-image-output-format", "gif"),
            ("x-muna-image-output-quality", "101"),
            ("x-muna-image-output-quality", "80"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(name, value.parse().unwrap());
            assert!(ImageEncoding::from_headers(&headers).is_err());
        }
        for json in [
            serde_json::json!({"dtype":"image", "data":[]}),
            serde_json::json!({"dtype":"image_list", "data":[{"dtype":"image_list", "data":[]}]}),
            serde_json::json!({"dtype":"image_list", "data":[{"dtype":"string", "data":"data:text/plain;base64,YQ=="}]}),
        ] {
            let remote = serde_json::from_value(json).unwrap();
            assert!(parse_remote_value(&remote).is_err());
        }
    }
}
