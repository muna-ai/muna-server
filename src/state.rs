/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use muna::Muna;

use crate::serving::dispatch::Dispatcher;
use crate::serving::registry::ModelRegistry;

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
    /// Edge indexer callback URLs for the KV relay, refreshed by every
    /// heartbeat response.
    pub event_callbacks: tokio::sync::watch::Sender<Vec<String>>,
}

pub(crate) struct AppState {
    /// Muna client.
    pub muna: Arc<Muna>,
    /// Per-model load-state machine.
    pub registry: ModelRegistry,
    /// Per-model prediction dispatcher.
    pub dispatcher: Dispatcher,
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
        muna: Arc<Muna>,
        pinned: Option<HashSet<String>>,
        node: Option<NodeContext>
    ) -> Self {
        Self {
            registry: ModelRegistry::new(muna.clone(), pinned),
            dispatcher: Dispatcher::new(muna.clone()),
            muna,
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
        let result = self
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
