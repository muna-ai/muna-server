/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use muna::MunaClient;

use crate::serving::cache::CacheTracker;
use crate::serving::dispatch::Dispatcher;
use crate::serving::registry::ModelRegistry;

/// Per-tag deployment keys delivered by control-plane residency directives
/// (`HeartbeatResponse::keys`). Read by the registry loader and the cache
/// downloader when constructing per-model Muna clients; tags without an
/// entry fall back to the process-wide `$MUNA_ACCESS_KEY`.
pub(crate) type KeyStore = Arc<DashMap<String, String>>;

/// Control-plane wiring, present only when the server runs as a fleet node.
pub(crate) struct NodeContext {
    /// Node identity assigned by the control plane at provision time.
    pub node_id: String,
    /// Control plane base URL.
    pub control_plane_url: String,
    /// Heartbeat cadence.
    pub heartbeat_interval: Duration,
    /// KV relay event-accumulation window. Shorter than the heartbeat:
    /// this bounds edge-index staleness (a block admitted right after a
    /// flush is unroutable for a full window), and an idle window sends
    /// no HTTP at all.
    pub kv_flush_interval: Duration,
}

pub(crate) struct AppState {
    /// Per-model load-state machine.
    pub registry: ModelRegistry,
    /// Cached-tier tracker (resources on disk, no engine).
    pub cache: CacheTracker,
    /// Per-model prediction dispatcher.
    pub dispatcher: Dispatcher,
    /// Per-tag deployment keys from residency directives; see [`KeyStore`].
    pub keys: KeyStore,
    /// Resource-cache directory (env-derived), for disk metrics. There is
    /// no process-wide Muna client: each loaded model owns its own (see
    /// `ReadyModel::muna`).
    pub cache_path: PathBuf,
    /// Control-plane wiring; `None` in standalone mode.
    pub node: Option<NodeContext>,
    /// Process start, for uptime reporting.
    pub start_time: Instant,
    /// Set by `/drain` or a heartbeat response; new inference requests are
    /// rejected while draining.
    draining: AtomicBool,
    /// Next API check-in time for each loaded model.
    runtime_checkins: tokio::sync::RwLock<BTreeMap<String, u64>>,
}

impl AppState {
    const CHECKIN_INTERVAL_SECONDS: u64 = 12 * 60 * 60;
    const CHECKIN_RETRY_SECONDS: u64 = 60 * 60;

    pub(crate) fn new(
        pinned: Option<HashSet<String>>,
        node: Option<NodeContext>
    ) -> Self {
        let keys: KeyStore = Arc::new(DashMap::new());
        // Throwaway client purely for the env-derived cache location; it
        // holds no credential and makes no requests.
        let cache_path = MunaClient::new(None, None).cache_path().to_path_buf();
        Self {
            registry: ModelRegistry::new(keys.clone(), pinned),
            cache: CacheTracker::new(keys.clone()),
            dispatcher: Dispatcher::new(),
            keys,
            cache_path,
            node,
            start_time: Instant::now(),
            draining: AtomicBool::new(false),
            runtime_checkins: tokio::sync::RwLock::new(BTreeMap::new()),
        }
    }

    pub(crate) fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Relaxed)
    }

    pub(crate) fn set_draining(&self, draining: bool) {
        self.draining.store(draining, Ordering::Relaxed);
    }

    /// Refresh the runtime token for a loaded model if its check-in is due.
    /// Network-only (prediction spec fetch); no native FFI involved.
    pub(crate) async fn check_in_if_due(&self, model: &str) {
        let now = unix_now();
        let next_checkin = self.runtime_checkins.read().await.get(model).copied();
        let Some(next_checkin) = next_checkin else {
            // The inference path performs the initial check-in while loading.
            return;
        };
        if now < next_checkin {
            return;
        }
        // Check in through the model's own (keyed) Muna instance; a model
        // that is no longer loaded has nothing to check in for.
        let Some(ready) = self.registry.ready(model) else {
            return;
        };
        let result = ready
            .muna
            .predictions
            .create(model, None, None, None, None)
            .await;
        let delay = if result.is_ok() {
            Self::CHECKIN_INTERVAL_SECONDS
        } else {
            Self::CHECKIN_RETRY_SECONDS
        };
        self.runtime_checkins
            .write()
            .await
            .insert(model.to_owned(), now.saturating_add(delay));
        if let Err(error) = result {
            // A loaded predictor remains usable offline; retry liveness later.
            tracing::warn!("failed to refresh runtime token for {model}: {error}");
        }
    }

    /// Schedule the first check-in for a freshly loaded model.
    pub(crate) async fn mark_model_loaded(&self, model: String) {
        let now = unix_now();
        self.runtime_checkins
            .write()
            .await
            .entry(model)
            .or_insert_with(|| now.saturating_add(Self::CHECKIN_INTERVAL_SECONDS));
    }
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
