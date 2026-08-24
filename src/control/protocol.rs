/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-control wire protocol (v2).
//!
//! Every type that crosses the node <-> control-plane boundary lives here;
//! the standalone control-plane project mirrors this module
//! (`control-plane/src/nodes/protocol.rs`) and pins it with golden-JSON
//! tests -- change BOTH ends together.
//!
//! Goal-vs-actual vocabulary split: `HeartbeatResponse` declares the GOAL
//! (three-valued `Residency`, stable), `NodeStatus` reports the ACTUAL
//! (five-valued `ModelLifecycle`, transitional). The node diffs the goal
//! against its own state locally and walks the warmth ladder
//! (`caching -> cached -> loading -> ready`) on its own; the plane watches
//! actuals progress each beat and never micromanages intermediate hops.

use std::sync::atomic::Ordering;

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::metrics::{collect_gpu_metrics, GpuMetrics};
use crate::serving::batch::BatchPlan;
use crate::serving::cache::{CacheState, CacheTracker};
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
    /// Warmth tier and counters for every known model, including models
    /// cached on disk with no engine loaded.
    pub models: Vec<ModelStatus>,
}

/// Lifecycle state of a model on this node -- the ACTUAL half of the
/// protocol. Serializes as a lowercase string on the wire (`"caching"`,
/// `"cached"`, `"loading"`, `"ready"`, `"failed"`).
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelLifecycle {
    /// Resource download in progress.
    Caching,
    /// Resources complete on disk; no engine loaded. The
    /// few-second-coldstart tier.
    Cached,
    /// Engine load in progress.
    Loading,
    /// Engine loaded and serving.
    Ready,
    /// Last load (or cache) attempt failed; see `ModelStatus::error`.
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
    /// Resource download progress percentage, when `state` is `caching`.
    /// Currently always absent: per-tag attribution needs the muna client
    /// to expose the resource list, which is a later muna-rs change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_pct: Option<u8>,
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

/// Desired residency for one model.
#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Residency {
    /// Engine loaded in the serving process (GPU memory for GPU models),
    /// serving. Implies disk. Named for the process, not the GPU: CPU-only
    /// models occupy this tier too.
    Process,
    /// Resources complete on disk; no engine. The few-second-coldstart tier.
    Disk,
    /// No residency guarantee: engine unloaded; disk eviction PERMITTED
    /// but not required (node-local GC under disk pressure decides --
    /// preserves the "resources are never deleted" behavior for now).
    None,
}

/// Goal descriptor for one model in a heartbeat response -- the GOAL
/// mirror of `ModelStatus` (the ACTUAL half): both sides of the protocol
/// are flat, tag-keyed lists.
#[derive(Deserialize)]
pub(crate) struct ModelDescriptor {
    /// Predictor tag.
    pub tag: String,
    /// Desired residency.
    pub residency: Residency,
    /// Download credential: the predictor-bound deployment key
    /// (`muna_sk_...`) from the tag's shared deployment, sent with
    /// `process`/`disk` directives. The node uses it to retrieve and
    /// download the predictor; absent falls back to the process-wide
    /// `MUNA_ACCESS_KEY` (standalone `muna deploy` behavior).
    #[serde(default)]
    pub key: Option<String>,
}

/// Control plane's reply to a heartbeat POST -- the declarative goal.
///
/// v1's imperative deltas (`load_models` / `unload_models`) and
/// `event_callback_urls` are GONE: the node diffs the goal residency map
/// against its own state, and the KV relay derives its single target from
/// the control-plane URL it already has (well-known `/v1/kv/events` route).
#[derive(Deserialize, Default)]
pub(crate) struct HeartbeatResponse {
    /// Goal descriptor (residency + download credential) per tag. An
    /// ABSENT tag means "no opinion" -- the node keeps its current state
    /// -- so an empty list (also the parse-failure / standalone default)
    /// is a guaranteed no-op. Fail-safe by construction: a cold-started or
    /// truncated plane response can never mass-unload a fleet.
    #[serde(default)]
    pub models: Vec<ModelDescriptor>,
    /// Stop accepting new inference requests.
    #[serde(default)]
    pub drain: bool,
}

/// Batch of KV events POSTed to the control plane's `/v1/kv/events` route.
#[derive(Serialize)]
pub(crate) struct RelayBatch<'a> {
    /// Node identity assigned by the control plane.
    pub worker_id: &'a str,
    /// Predictor tag the events belong to.
    pub model: &'a str,
    /// Publisher epoch (32-hex); fresh per predictor instantiation. The
    /// indexer drops all state for a worker on epoch change.
    pub epoch: &'a str,
    /// Inclusive `(first, last)` publisher seq covered by `events`, for
    /// indexer-side gap detection.
    pub seq_range: (u64, u64),
    /// Whether this batch begins with a full-state snapshot; the indexer
    /// applies it as reset-then-set.
    pub snapshot: bool,
    /// Events verbatim from the LLM engine stream, in publish order.
    pub events: &'a [JsonValue],
}

/// The control plane's reply to a `RelayBatch` POST.
#[derive(Deserialize, Default)]
pub(crate) struct EdgeResponse {
    /// The indexer lost continuity (unknown worker/epoch or a seq gap) and
    /// needs a snapshot before applying further deltas.
    #[serde(default)]
    pub need_snapshot: bool,
}

impl NodeStatus {

    pub(crate) fn collect(state: &AppState) -> NodeStatus {
        let disk = crate::metrics::disk_space_mb(&state.cache_path);
        NodeStatus {
            node_id: state.node.as_ref().map(|n| n.node_id.clone()),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_s: state.start_time.elapsed().as_secs(),
            draining: state.is_draining(),
            disk_free_mb: disk.map(|(free, _)| free),
            disk_total_mb: disk.map(|(_, total)| total),
            gpus: collect_gpu_metrics(),
            models: collect_model_status(&state.registry, &state.cache),
        }
    }
}

/// Merge the engine registry (loading/ready/failed) with the cache tracker
/// (caching/cached/failed): the registry wins for tags it knows -- a loaded
/// model is cached by definition, and reporting the higher tier is what the
/// plane's placement wants to see.
fn collect_model_status(
    registry: &ModelRegistry,
    cache: &CacheTracker
) -> Vec<ModelStatus> {
    let mut statuses: Vec<ModelStatus> = registry
        .snapshot()
        .into_iter()
        .map(|(tag, state)| match state {
            ModelState::Loading { .. } => empty_status(tag, ModelLifecycle::Loading, None),
            ModelState::Ready(model) => ModelStatus {
                tag,
                state: ModelLifecycle::Ready,
                error: None,
                batch_mode: Some((&model.plan).into()),
                progress_pct: None,
                queue_depth: model.stats.queue_depth.load(Ordering::Relaxed),
                total_predictions: model.stats.total_predictions.load(Ordering::Relaxed),
                avg_latency_ms: model.stats.avg_latency_ms(),
                load_time_ms: model.stats.load_time_ms(),
                vram_mb: model.stats.vram_mb(),
            },
            ModelState::Failed { error, .. } => {
                empty_status(tag, ModelLifecycle::Failed, Some(error))
            }
        })
        .collect();
    for (tag, state) in cache.snapshot() {
        if statuses.iter().any(|status| status.tag == tag) {
            continue;
        }
        let status = match state {
            CacheState::Caching => empty_status(tag, ModelLifecycle::Caching, None),
            CacheState::Cached => empty_status(tag, ModelLifecycle::Cached, None),
            CacheState::Failed { error, .. } => {
                empty_status(tag, ModelLifecycle::Failed, Some(error))
            }
        };
        statuses.push(status);
    }
    statuses
}

fn empty_status(
    tag: String,
    state: ModelLifecycle,
    error: Option<String>
) -> ModelStatus {
    ModelStatus {
        tag,
        state,
        error,
        batch_mode: None,
        progress_pct: None,
        queue_depth: 0,
        total_predictions: 0,
        avg_latency_ms: 0.0,
        load_time_ms: 0.0,
        vram_mb: None,
    }
}
