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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::notifications::NotificationCenter;
use crate::serving::prepared::PreparedModels;
use crate::serving::preparation::{FailureStage, PreparationFailure, PreparationProgress, PreparationTracker};
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
    /// This deployment has no implementation compatible with the node.
    /// Rechecked only after retirement or a changed deployment key.
    Ineligible { error: String },
    /// The last cache attempt failed; retried after backoff.
    Failed { error: String, at: Instant },
}

/// Resource queue priority for this model's preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachePriority {
    /// Process-tier resources take precedence over queued prefetch files.
    Immediate,
    /// Disk-only resources use the background queue priority.
    Prefetch,
}

/// Download delegate: localizes one tag's resources on disk. Injectable so
/// lifecycle behavior is testable without the API.
type Downloader = Arc<
    dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<(), String>>
        + Send
        + Sync
>;

#[derive(Clone)]
pub(crate) struct CacheEntry {
    pub state: CacheState,
    pub key_hash: Option<String>,
}

/// Tracks which tags have complete resources on disk. Cloneable handle;
/// state is shared.
#[derive(Clone)]
pub(crate) struct CacheTracker {
    keys: KeyStore,
    /// Performs the download for one tag.
    downloader: Downloader,
    preparation: PreparationTracker,
    prepared: PreparedModels,
    /// Per-tag disk state. Absent-from-map means never requested (or
    /// retired by the control plane); shared with the download tasks, which
    /// write the terminal `Cached` / `Ineligible` / `Failed` state on completion.
    states: Arc<DashMap<String, CacheEntry>>,
    /// Serializes task completion with retirement so late downloads cannot reannounce tags.
    downloads: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    /// Poked on every tier write so the heartbeat reports it at once.
    notifications: Arc<NotificationCenter>,
}

impl CacheTracker {
    pub(crate) fn disk_changed(&self) { self.prepared.disk_changed(); }


    pub(crate) fn new(
        keys: KeyStore,
        notifications: Arc<NotificationCenter>,
        preparation: PreparationTracker,
        prepared: PreparedModels
    ) -> Self {
        let identities = keys.clone();
        let progress = preparation.clone();
        let models = prepared.clone();
        let downloader: Downloader = Arc::new(move |tag| {
            let key = keys.get(&tag).map(|entry| entry.value().clone());
            let progress = progress.clone();
            let models = models.clone();
            Box::pin(async move { models.prepare(&tag, key, progress).await.map(|_| ()) })
        });
        let mut tracker = Self::with_downloader(downloader, notifications);
        tracker.keys = identities;
        tracker.preparation = preparation;
        tracker.prepared = prepared;
        tracker
    }

    fn with_downloader(
        downloader: Downloader,
        notifications: Arc<NotificationCenter>
    ) -> Self {
        Self {
            keys: Arc::default(),
            downloader,
            preparation: PreparationTracker::new(notifications.clone()),
            prepared: PreparedModels::default(),
            states: Arc::new(DashMap::new()),
            downloads: Arc::default(),
            notifications
        }
    }

    /// Ensure the tag's resources are on disk. Idempotent and single-flight:
    /// a tag already caching or cached is a no-op; a failed tag retries
    /// only after backoff. Both resolve metadata immediately; individual files
    /// are ordered by the shared resource scheduler.
    pub(crate) fn ensure_cached(&self, tag: &str, priority: CachePriority) {
        self.prepared.set_process(tag, priority == CachePriority::Immediate);
        let mut downloads = self.downloads.lock().unwrap();
        let key_hash = self.keys.get(tag).map(|key| crate::client::key_hash(key.value()));
        match self.states.entry(tag.to_string()) {
            dashmap::Entry::Occupied(mut entry) => {
                match &entry.get().state {
                    CacheState::Caching | CacheState::Cached | CacheState::Ineligible { .. } => return,
                    CacheState::Failed { at, .. } => {
                        if self.preparation.failure(tag).is_some_and(|failure| !failure.retrying) { return; }
                        if at.elapsed() < FAILED_RETRY_BACKOFF {
                            return;
                        }
                        entry.insert(CacheEntry { state: CacheState::Caching, key_hash: key_hash.clone() });
                    }
                }
            }
            dashmap::Entry::Vacant(entry) => {
                entry.insert(CacheEntry { state: CacheState::Caching, key_hash: key_hash.clone() });
            }
        }
        self.preparation.begin(tag);
        self.notifications.status_changed();
        let task = self.spawn_download(tag.to_string(), key_hash);
        downloads.insert(tag.to_string(), task);
    }

    /// Stop advertising a tag without deleting its content-addressed files.
    pub(crate) fn forget(&self, tag: &str) {
        let mut downloads = self.downloads.lock().unwrap();
        if let Some(task) = downloads.remove(tag) {
            task.abort();
        }
        self.states.remove(tag);
        self.prepared.forget(tag);
        self.preparation.forget(tag);
        self.notifications.status_changed();
    }

    /// Snapshot every tracked tag for status reporting.
    pub(crate) fn snapshot(&self) -> Vec<(String, CacheEntry)> {
        self.states
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    pub(crate) fn requirements(&self, tag: &str) -> Option<super::preparation::ModelRequirements> {
        self.preparation.requirements(tag)
    }

    pub(crate) fn build(&self, tag: &str) -> Option<String> {
        self.preparation.build(tag)
    }

    pub(crate) fn progress(&self, tag: &str) -> Option<PreparationProgress> {
        self.preparation.snapshot(tag)
    }

    pub(crate) fn failure(&self, tag: &str) -> Option<PreparationFailure> {
        self.preparation.failure(tag)
    }

    fn spawn_download(&self, tag: String, key_hash: Option<String>) -> tokio::task::JoinHandle<()> {
        let downloads = self.downloads.clone();
        let downloader = self.downloader.clone();
        let states = self.states.clone();
        let notifications = self.notifications.clone();
        let progress = self.preparation.scoped(&tag);
        // Capture the credential before scheduling; changing the goal later
        // must not pair a new key's resources with the previous attempt's hash.
        let downloading = downloader(tag.clone());
        tokio::spawn(async move {
            let start = Instant::now();
            let result = downloading.await;
            let mut downloads = downloads.lock().unwrap();
            let state = match result {
                Ok(()) => {
                    progress.cached(&tag);
                    tracing::info!(
                        tag = %tag,
                        elapsed_ms = %format!("{:.0}", start.elapsed().as_secs_f64() * 1000.0),
                        "model cached on disk"
                    );
                    CacheState::Cached
                }
                Err(error) => {
                    if progress.failure(&tag).is_some_and(|failure| failure.is_ineligible()) {
                        tracing::info!(tag = %tag, "model is ineligible on this node");
                        CacheState::Ineligible { error }
                    } else {
                        tracing::warn!(tag = %tag, error = %error, "cache download failed");
                        if progress.failure(&tag).is_none() { progress.failed(&tag, FailureStage::Resolve); }
                        CacheState::Failed { error, at: Instant::now() }
                    }
                }
            };
            downloads.remove(&tag);
            states.insert(tag, CacheEntry { state, key_hash });
            notifications.status_changed();
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A tracker whose downloads succeed instantly, counting each start.
    fn tracker(
        starts: Arc<AtomicUsize>
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
            Arc::new(NotificationCenter::default())
        )
    }

    async fn wait_until(
        tracker: &CacheTracker,
        tag: &str,
        predicate: impl Fn(&CacheState) -> bool
    ) -> bool {
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if tracker.states.get(tag).is_some_and(|state| predicate(&state.state)) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        false
    }

    #[tokio::test]
    async fn ineligible_deployment_is_memoized_until_its_key_changes() {
        let starts = Arc::new(AtomicUsize::new(0));
        let notifications = Arc::new(NotificationCenter::default());
        let preparation = PreparationTracker::new(notifications.clone());
        let downloader: Downloader = {
            let starts = starts.clone();
            let preparation = preparation.clone();
            Arc::new(move |tag| {
                let starts = starts.clone();
                let preparation = preparation.clone();
                Box::pin(async move {
                    if starts.fetch_add(1, Ordering::SeqCst) == 0 {
                        preparation.rejected(&tag, PreparationFailure::incompatible());
                        return Err("No active implementation for this architecture".into());
                    }
                    Ok(())
                })
            })
        };
        let mut tracker = CacheTracker::with_downloader(downloader, notifications);
        tracker.preparation = preparation;
        tracker.keys.insert("@a/x".into(), "old-key".into());
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Ineligible { .. })).await);
        assert_eq!(tracker.snapshot()[0].1.key_hash, Some(crate::client::key_hash("old-key")));
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(matches!(tracker.snapshot()[0].1.state, CacheState::Ineligible { .. }));
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert!(!tracker.failure("@a/x").unwrap().retrying);

        // The heartbeat retires preparation when it receives a new deployment key.
        tracker.forget("@a/x");
        tracker.keys.insert("@a/x".into(), "new-key".into());
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Cached)).await);
        assert_eq!(tracker.snapshot()[0].1.key_hash, Some(crate::client::key_hash("new-key")));
        assert!(tracker.failure("@a/x").is_none());
        assert_eq!(starts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn download_errors_remain_failures() {
        let downloader: Downloader = Arc::new(|_| Box::pin(async { Err("HTTP 503".into()) }));
        let tracker = CacheTracker::with_downloader(downloader, Arc::new(NotificationCenter::default()));
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Failed { .. })).await);
        assert_eq!(tracker.failure("@a/x").unwrap().code, "resource_resolution_failed");
    }

    #[tokio::test]
    async fn forgetting_cached_tag_keeps_files_and_allows_readmission() {
        let file = std::env::temp_dir().join(format!("muna-cache-retirement-{}", std::process::id()));
        std::fs::write(&file, b"cached resource").unwrap();
        let starts = Arc::new(AtomicUsize::new(0));
        let tracker = tracker(starts.clone());
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Cached)).await);
        tracker.preparation.manifest("@a/x", file.parent().unwrap(), &[
            muna::types::PredictionResource {
                kind: "dso".into(),
                url: format!("https://cdn.example/resources/{}", file.file_name().unwrap().to_str().unwrap()),
                name: None,
                size: Some(15),
            }
        ]).await;
        assert!(tracker.build("@a/x").is_some());
        assert_eq!(tracker.progress("@a/x").unwrap().resources_completed, 1);
        tracker.forget("@a/x");
        assert!(tracker.snapshot().is_empty());
        assert!(tracker.progress("@a/x").is_none());
        assert!(tracker.build("@a/x").is_none());
        assert_eq!(std::fs::read(&file).unwrap(), b"cached resource");
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Cached)).await);
        assert_eq!(starts.load(Ordering::SeqCst), 2);
        std::fs::remove_file(file).unwrap();
    }

    #[tokio::test]
    async fn forgetting_inflight_download_cancels_completion() {
        let started = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(tokio::sync::Notify::new());
        let downloader: Downloader = {
            let started = started.clone();
            let completed = completed.clone();
            let release = release.clone();
            Arc::new(move |_| {
                let started = started.clone();
                let completed = completed.clone();
                let release = release.clone();
                Box::pin(async move {
                    started.notify_one();
                    release.notified().await;
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            })
        };
        let tracker = CacheTracker::with_downloader(downloader, Arc::new(NotificationCenter::default()));
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        started.notified().await;
        tracker.forget("@a/x");
        release.notify_one();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(tracker.snapshot().is_empty());
        assert_eq!(completed.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn ensure_cached_is_single_flight() {
        let starts = Arc::new(AtomicUsize::new(0));
        let tracker = tracker(starts.clone());
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/x", |s| matches!(s, CacheState::Cached)).await);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cache_observations_keep_their_attempts_deployment_fingerprint() {
        let starts = Arc::new(AtomicUsize::new(0));
        let tracker = tracker(starts.clone());
        tracker.keys.insert("@a/x".into(), "old-key".into());
        tracker.ensure_cached("@a/x", CachePriority::Immediate);
        assert!(wait_until(&tracker, "@a/x", |state| matches!(state, CacheState::Cached)).await);
        tracker.keys.insert("@a/x".into(), "new-key".into());
        assert_eq!(tracker.snapshot()[0].1.key_hash, Some(crate::client::key_hash("old-key")));
        tracker.forget("@a/x");
        tracker.ensure_cached("@a/x", CachePriority::Prefetch);
        assert!(wait_until(&tracker, "@a/x", |state| matches!(state, CacheState::Cached)).await);
        assert_eq!(tracker.snapshot()[0].1.key_hash, Some(crate::client::key_hash("new-key")));
        assert_eq!(starts.load(Ordering::SeqCst), 2);
    }

}
