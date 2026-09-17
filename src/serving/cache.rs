/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Cached-tier tracker: the disk half of the warmth ladder.
//!
//! A model is *cached* when its resources are complete on disk with no
//! engine loaded -- the few-second-coldstart tier the control plane
//! prepositions with `Residency::Disk` goals. Caching a tag reuses the
//! muna client's download-only prediction (empty inputs map): it resolves
//! the tag's resource list and downloads what is missing, so re-validating
//! an already-cached tag reduces to statting files. The tracker memoizes
//! the outcome and reports it in every heartbeat -- the plane believes
//! reports and never assumes a download succeeded (the re-announcement
//! principle).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Value};
use muna::Muna;

use crate::client::ServerClient;
use crate::notifications::NotificationCenter;
use crate::serving::download_gate::DownloadGate;
use crate::serving::predict;
use crate::state::KeyStore;

/// A failed cache attempt is not retried until this backoff elapses, so a
/// plane re-asserting `disk` every beat cannot hot-loop downloads.
const FAILED_RETRY_BACKOFF: Duration = Duration::from_secs(60);

/// Disk state of one tag.
#[derive(Clone)]
pub(crate) enum CacheState {
    /// Resource download in progress.
    Caching,
    /// Resources complete on disk.
    Cached,
    /// The last cache attempt failed; retried after backoff.
    Failed { error: String, at: Instant },
}

/// Whether a cache request may wait behind in-flight process-tier loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachePriority {
    /// Process-tier tag: the engine load is already fetching these files;
    /// start now and let the client's per-path single-flight de-duplicate.
    Immediate,
    /// Disk-tier prefetch: wait until no process-tier download is in
    /// flight, so it does not share the disk with a load the plane is
    /// waiting on (see `download_gate.rs`).
    Prefetch,
}

/// Download delegate: localizes one tag's resources on disk. Injectable so
/// gate ordering is testable without the API.
type Downloader = Arc<
    dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<(), String>>
        + Send
        + Sync
>;

/// Tracks which tags have complete resources on disk. Cloneable handle;
/// state is shared.
#[derive(Clone)]
pub(crate) struct CacheTracker {
    /// Performs the download for one tag.
    downloader: Downloader,
    /// Waiting side of the process-tier-first download ordering.
    download_gate: Arc<DownloadGate>,
    /// Per-tag disk state. Absent-from-map means never requested (or
    /// forgotten after a failure); shared with the download tasks, which
    /// write the terminal `Cached` / `Failed` state on completion.
    states: Arc<DashMap<String, CacheState>>,
    /// Poked on every tier write so the heartbeat reports it at once.
    notifications: Arc<NotificationCenter>,
}

impl CacheTracker {

    pub(crate) fn new(
        keys: KeyStore,
        notifications: Arc<NotificationCenter>,
        download_gate: Arc<DownloadGate>
    ) -> Self {
        let downloader: Downloader = Arc::new(move |tag| {
            let key = keys.get(&tag).map(|entry| entry.value().clone());
            Box::pin(async move { download_resources(tag, key).await })
        });
        Self::with_downloader(downloader, notifications, download_gate)
    }

    fn with_downloader(
        downloader: Downloader,
        notifications: Arc<NotificationCenter>,
        download_gate: Arc<DownloadGate>
    ) -> Self {
        Self {
            downloader,
            download_gate,
            states: Arc::new(DashMap::new()),
            notifications
        }
    }

    /// Ensure the tag's resources are on disk. Idempotent and single-flight:
    /// a tag already caching or cached is a no-op; a failed tag retries
    /// only after backoff. `Prefetch` requests queue behind in-flight
    /// process-tier downloads; `Immediate` ones start at once.
    pub(crate) fn ensure_cached(&self, tag: &str, priority: CachePriority) {
        match self.states.entry(tag.to_string()) {
            dashmap::Entry::Occupied(mut entry) => {
                match entry.get() {
                    CacheState::Caching | CacheState::Cached => return,
                    CacheState::Failed { at, .. } => {
                        if at.elapsed() < FAILED_RETRY_BACKOFF {
                            return;
                        }
                        entry.insert(CacheState::Caching);
                    }
                }
            }
            dashmap::Entry::Vacant(entry) => {
                entry.insert(CacheState::Caching);
            }
        }
        self.notifications.status_changed();
        self.spawn_download(tag.to_string(), priority);
    }

    /// Drop a failed cache record (nothing was achieved on disk, so there
    /// is nothing truthful to keep reporting). `Cached` records are kept:
    /// the resources genuinely are on disk, and the plane's placement wants
    /// to see that even under a `none` goal.
    pub(crate) fn forget_failed(&self, tag: &str) {
        let removed = self.states
            .remove_if(tag, |_, state| matches!(state, CacheState::Failed { .. }));
        if removed.is_some() {
            self.notifications.status_changed();
        }
    }

    /// Snapshot every tracked tag for status reporting.
    pub(crate) fn snapshot(&self) -> Vec<(String, CacheState)> {
        self.states
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    fn spawn_download(&self, tag: String, priority: CachePriority) {
        let downloader = self.downloader.clone();
        let states = self.states.clone();
        let notifications = self.notifications.clone();
        let gate = self.download_gate.clone();
        tokio::spawn(async move {
            if priority == CachePriority::Prefetch {
                let queued = Instant::now();
                gate.wait_idle().await;
                if queued.elapsed() > Duration::from_secs(1) {
                    tracing::info!(
                        tag = %tag,
                        waited_ms = %format!("{:.0}", queued.elapsed().as_secs_f64() * 1000.0),
                        "prefetch waited for process-tier downloads"
                    );
                }
            }
            let start = Instant::now();
            let state = match downloader(tag.clone()).await {
                Ok(()) => {
                    tracing::info!(
                        tag = %tag,
                        elapsed_ms = %format!("{:.0}", start.elapsed().as_secs_f64() * 1000.0),
                        "model cached on disk"
                    );
                    CacheState::Cached
                }
                Err(error) => {
                    tracing::warn!(tag = %tag, error = %error, "cache download failed");
                    CacheState::Failed { error, at: Instant::now() }
                }
            };
            states.insert(tag, state);
            notifications.status_changed();
        });
    }
}

/// Localize a tag's resources through the download-only prediction.
///
/// Ephemeral keyed instance per download: the download-only prediction
/// localizes resources without loading a native predictor, so no handle
/// outlives this call (unlike the registry's persistent per-model instance).
async fn download_resources(
    tag: String,
    key: Option<String>
) -> Result<(), String> {
    let muna = Arc::new(Muna::with_client(Arc::new(ServerClient::with_key(key))));
    // Download-only convention: an empty (but present) inputs map makes the
    // muna client create a raw prediction and localize its resources without
    // loading any engine. Acceleration must be a LOCAL flavor: without it
    // the API resolves the tag as a remote predictor, which compiled models
    // do not have.
    let prediction = predict::run(None, move || async move {
        muna.predictions.create(
            &tag,
            Some(HashMap::<String, Value>::new()),
            Some(Acceleration::LocalAuto),
            None,
            None
        ).await
    }).await.map_err(|e| e.to_string())?;
    match prediction.error {
        Some(error) => Err(error),
        None => Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A tracker whose downloads succeed instantly, counting each start.
    fn tracker(
        starts: Arc<AtomicUsize>,
        gate: Arc<DownloadGate>
    ) -> CacheTracker {
        let downloader: Downloader = Arc::new(move |_tag| {
            let starts = starts.clone();
            Box::pin(async move {
                starts.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        });
        CacheTracker::with_downloader(
            downloader,
            Arc::new(NotificationCenter::default()),
            gate
        )
    }

    async fn wait_until(
        tracker: &CacheTracker,
        tag: &str,
        predicate: impl Fn(&CacheState) -> bool
    ) -> bool {
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if tracker.states.get(tag).is_some_and(|state| predicate(state.value())) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        false
    }

    #[tokio::test]
    async fn prefetch_waits_for_priority_downloads() {
        let starts = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(DownloadGate::new());
        let tracker = tracker(starts.clone(), gate.clone());
        let priority = gate.enter_priority();
        tracker.ensure_cached("@a/prefetch", CachePriority::Prefetch);
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Queued: reported as caching, download not started.
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        assert!(matches!(
            tracker.states.get("@a/prefetch").unwrap().value(),
            CacheState::Caching
        ));
        drop(priority);
        assert!(wait_until(&tracker, "@a/prefetch", |s| matches!(s, CacheState::Cached)).await);
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn immediate_requests_bypass_the_gate() {
        let starts = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(DownloadGate::new());
        let tracker = tracker(starts.clone(), gate.clone());
        let _priority = gate.enter_priority();
        tracker.ensure_cached("@a/process", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/process", |s| matches!(s, CacheState::Cached)).await);
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ensure_cached_is_single_flight() {
        let starts = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(DownloadGate::new());
        let tracker = tracker(starts.clone(), gate);
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Cached)).await);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }
}
