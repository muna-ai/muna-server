/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-control wire protocol.
//!
//! Every type that crosses the node <-> control-plane (or node -> edge
//! indexer) boundary lives here; the standalone control-plane project copies
//! or depends on this module.

use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::metrics::{collect_gpu_metrics, GpuMetrics};
use crate::serving::batch::BatchPlan;
use crate::serving::registry::{ModelRegistry, ModelState};
use crate::state::AppState;

#[derive(Serialize)]
pub(crate) struct NodeStatus {
    /// Node identity assigned by the control plane; absent in standalone mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    /// muna-server version.
    pub version: String,
    /// Seconds since the server process started.
    pub uptime_s: u64,
    /// Whether the node is rejecting new inference requests.
    pub draining: bool,
    /// Free space in MB on the resource-cache volume. Resources are never
    /// deleted, so this only shrinks; the control plane uses it to stop
    /// placing models on a full node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_free_mb: Option<u64>,
    /// Total space in MB on the data volume.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disk_total_mb: Option<u64>,
    /// Per-device GPU metrics (system RAM fallback on CPU-only nodes).
    pub gpus: Vec<GpuMetrics>,
    /// Warmth tier and counters for every known model.
    pub models: Vec<ModelStatus>,
}

/// Lifecycle state of a model on this node. Serializes as a lowercase
/// string on the wire (`"loading"`, `"ready"`, `"failed"`).
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelLifecycle {
    /// Model load in progress.
    Loading,
    /// Model loaded and serving.
    Ready,
    /// Last load attempt failed; see `ModelStatus::error`.
    Failed,
}

/// How the node dispatches predictions for a loaded model, derived from
/// the predictor signature's batch config. Serializes as a lowercase string
/// on the wire (`"sequential"`, `"buffered"`, `"continuous"`). Distinct from
/// `muna::types::BatchMode`: signature-level static and dynamic both dispatch
/// as `Buffered` (the server never pads a partial batch).
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum DispatchMode {
    /// One prediction at a time, behind a per-model mutex.
    Sequential,
    /// Requests merged up to capacity or deadline, then split per caller.
    Buffered,
    /// Submitted concurrently; the compiled engine batches internally.
    Continuous,
}

impl From<&BatchPlan> for DispatchMode {
    fn from(plan: &BatchPlan) -> Self {
        match plan {
            BatchPlan::Sequential      => Self::Sequential,
            BatchPlan::Buffered { .. } => Self::Buffered,
            BatchPlan::Continuous      => Self::Continuous,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ModelStatus {
    /// Predictor tag.
    pub tag: String,
    /// Model lifecycle state.
    pub state: ModelLifecycle,
    /// Load failure message, when `state` is `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Dispatch mode, when `state` is `ready`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_mode: Option<DispatchMode>,
    /// Number of predictions currently waiting in the model's dispatch queue.
    pub queue_depth: u32,
    /// Total predictions made.
    pub total_predictions: u64,
    /// Average prediction latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Time it took to download and load the model in milliseconds.
    pub load_time_ms: f64,
    /// Estimated VRAM used by this model in MB (measured at load time).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_mb: Option<u64>,
}

/// Control plane's reply to a heartbeat POST. Reconciliation is driven by
/// imperative deltas: `load_models` / `unload_models` are idempotent no-ops
/// for already-satisfied tags, and the control plane self-heals by diffing
/// against the fresh status every beat.
#[derive(Deserialize, Default)]
pub(crate) struct HeartbeatResponse {
    /// Tags to warm (sentinel prediction through the registry).
    #[serde(default)]
    pub load_models: Vec<String>,
    /// Tags to unload. Removes the engine only; downloaded resources are
    /// never deleted and persist on disk.
    #[serde(default)]
    pub unload_models: Vec<String>,
    /// Edge indexer callbacks for the KV relay; refreshed every beat.
    #[serde(default)]
    pub event_callback_urls: Vec<String>,
    /// Stop accepting new inference requests.
    #[serde(default)]
    pub drain: bool,
}

/// Batch of KV events POSTed to each edge indexer.
#[derive(Serialize)]
pub(crate) struct RelayBatch<'a> {
    /// Node identity assigned by the control plane.
    pub worker_id: &'a str,
    /// Predictor tag the events belong to.
    pub model: &'a str,
    /// Publisher epoch (32-hex); fresh per predictor instantiation. Edges
    /// drop all state for a worker on epoch change.
    pub epoch: &'a str,
    /// Inclusive `(first, last)` publisher seq covered by `events`, for
    /// edge-side gap detection.
    pub seq_range: (u64, u64),
    /// Whether this batch begins with a full-state snapshot; edges apply it
    /// as reset-then-set.
    pub snapshot: bool,
    /// Events verbatim from the LLM engine stream, in publish order.
    pub events: &'a [JsonValue],
}

/// Edge indexer's reply to a `RelayBatch` POST.
#[derive(Deserialize, Default)]
pub(crate) struct EdgeResponse {
    /// The edge lost continuity (unknown worker/epoch or a seq gap) and
    /// needs a snapshot before applying further deltas.
    #[serde(default)]
    pub need_snapshot: bool,
}

impl NodeStatus {

    pub(crate) fn collect(state: &AppState) -> NodeStatus {
        let disk = crate::metrics::disk_space_mb(state.muna.client.cache_path());
        NodeStatus {
            node_id: state.node.as_ref().map(|n| n.node_id.clone()),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_s: state.start_time.elapsed().as_secs(),
            draining: state.is_draining(),
            disk_free_mb: disk.map(|(free, _)| free),
            disk_total_mb: disk.map(|(_, total)| total),
            gpus: collect_gpu_metrics(),
            models: collect_model_status(&state.registry),
        }
    }
}

fn collect_model_status(registry: &ModelRegistry) -> Vec<ModelStatus> {
    registry
        .snapshot()
        .into_iter()
        .map(|(tag, state)| match state {
            ModelState::Loading { .. } => ModelStatus {
                tag,
                state: ModelLifecycle::Loading,
                error: None,
                batch_mode: None,
                queue_depth: 0,
                total_predictions: 0,
                avg_latency_ms: 0.0,
                load_time_ms: 0.0,
                vram_mb: None,
            },
            ModelState::Ready(model) => ModelStatus {
                tag,
                state: ModelLifecycle::Ready,
                error: None,
                batch_mode: Some((&model.plan).into()),
                queue_depth: model.stats.queue_depth.load(Ordering::Relaxed),
                total_predictions: model.stats.total_predictions.load(Ordering::Relaxed),
                avg_latency_ms: model.stats.avg_latency_ms(),
                load_time_ms: model.stats.load_time_ms(),
                vram_mb: model.stats.vram_mb(),
            },
            ModelState::Failed { error, .. } => ModelStatus {
                tag,
                state: ModelLifecycle::Failed,
                error: Some(error),
                batch_mode: None,
                queue_depth: 0,
                total_predictions: 0,
                avg_latency_ms: 0.0,
                load_time_ms: 0.0,
                vram_mb: None,
            },
        })
        .collect()
}
