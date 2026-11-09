/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-local preparation facts. Resource observations are shared by cache path;
//! each tag keeps its own complete manifest, never another tag's denominator.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use muna::types::PredictionResource;
use serde::{Deserialize, Serialize};

use crate::notifications::NotificationCenter;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);
const MAX_ACTIVE_RESOURCES: usize = 4;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PreparationStage {
    Resolving,
    Downloading,
    Cached,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureStage {
    Resolve,
    Download,
    Load,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PreparationFailure {
    pub stage: FailureStage,
    pub code: String,
    pub message: String,
    pub retrying: bool,
}

impl PreparationFailure {

    pub(crate) fn new(stage: FailureStage) -> Self {
        let (code, message) = match stage {
            FailureStage::Resolve => (
                "resource_resolution_failed",
                "Could not resolve predictor resources.",
            ),
            FailureStage::Download => (
                "resource_download_failed",
                "Could not download predictor resources.",
            ),
            FailureStage::Load => ("predictor_load_failed", "Could not load the predictor."),
        };
        // Detailed SDK/native errors remain in node logs and the existing error
        // field. Public summaries must not contain signed URLs or credentials.
        Self {
            stage,
            code: code.into(),
            message: message.into(),
            retrying: true,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ActiveResource {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub bytes_available: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct PreparationProgress {
    pub stage: PreparationStage,
    pub resources_completed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources_total: Option<u64>,
    pub bytes_available: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub active_resources: Vec<ActiveResource>,
}

impl PreparationProgress {

    pub(crate) fn percent(&self) -> Option<u8> {
        if self.stage == PreparationStage::Cached {
            return Some(100);
        }
        self.bytes_total.filter(|total| *total > 0).map(|total| {
            // In-flight bytes are not yet a published complete disk copy.
            ((u128::from(self.bytes_available) * 100 / u128::from(total)).min(99)) as u8
        })
    }
}

#[derive(Default)]
struct Resource {
    available: u64,
    total: Option<u64>,
    complete: bool,
    active: bool,
}

struct ManifestResource {
    kind: String,
    id: String,
    name: Option<String>,
    path: PathBuf,
    resource: Arc<Mutex<Resource>>,
}

struct Attempt {
    stage: PreparationStage,
    manifest: Option<Vec<ManifestResource>>,
    failure: Option<PreparationFailure>,
}

#[derive(Default)]
struct Tracking {
    resources: BTreeMap<PathBuf, Arc<Mutex<Resource>>>,
    attempts: BTreeMap<String, Attempt>,
    last_progress: Option<Instant>,
}

#[derive(Clone)]
pub(crate) struct PreparationTracker {
    tracking: Arc<Mutex<Tracking>>,
    notifications: Arc<NotificationCenter>,
}

impl PreparationTracker {

    pub(crate) fn new(notifications: Arc<NotificationCenter>) -> Self {
        Self {
            tracking: Arc::default(),
            notifications,
        }
    }

    pub(crate) fn begin(&self, tag: &str) {
        let mut tracking = self.tracking.lock().unwrap();
        // Cache and process-tier clients may resolve the same tag concurrently.
        tracking.attempts.entry(tag.into()).or_insert(Attempt {
            stage: PreparationStage::Resolving,
            manifest: None,
            failure: None,
        });
    }

    pub(crate) async fn manifest(&self, tag: &str, root: &Path, resources: &[PredictionResource]) {
        let mut manifest = Vec::new();
        let mut seen = HashSet::new();
        for source in resources {
            let (id, path) = resource_path(root, source);
            if !seen.insert(path.clone()) {
                continue;
            }
            let metadata = tokio::fs::metadata(&path)
                .await
                .ok()
                .filter(|meta| meta.is_file());
            let resource = {
                let mut tracking = self.tracking.lock().unwrap();
                tracking.resources.entry(path.clone()).or_default().clone()
            };
            {
                let mut value = resource.lock().unwrap();
                if let Some(metadata) = metadata {
                    value.available = metadata.len();
                    value.total = Some(metadata.len());
                    value.complete = true;
                    value.active = false;
                } else if !value.active && !value.complete {
                    value.available = 0;
                    value.total = source.size;
                    value.complete = false;
                }
            }
            manifest.push(ManifestResource {
                kind: source.kind.clone(),
                id,
                name: source.name.clone(),
                path,
                resource,
            });
        }
        let complete = manifest
            .iter()
            .all(|item| item.resource.lock().unwrap().complete);
        self.tracking.lock().unwrap().attempts.insert(
            tag.into(),
            Attempt {
                stage: if complete {
                    PreparationStage::Cached
                } else {
                    PreparationStage::Downloading
                },
                manifest: Some(manifest),
                failure: None,
            },
        );
        self.notifications.status_changed();
    }

    pub(crate) fn started(&self, path: &Path) {
        let resource = self
            .tracking
            .lock()
            .unwrap()
            .resources
            .entry(path.into())
            .or_default()
            .clone();
        let mut value = resource.lock().unwrap();
        value.available = 0;
        value.complete = false;
        value.active = true;
    }

    pub(crate) fn advance(
        &self,
        path: &Path,
        increment: u64,
        total: Option<u64>
    ) {
        let mut tracking = self.tracking.lock().unwrap();
        if let Some(resource) = tracking.resources.get(path) {
            let mut value = resource.lock().unwrap();
            value.total = total.or(value.total);
            value.available = value.available.saturating_add(increment);
            if let Some(total) = value.total {
                value.available = value.available.min(total);
            }
        }
        self.notify_progress(&mut tracking);
    }

    fn notify_progress(&self, tracking: &mut Tracking) {
        if tracking
            .last_progress
            .is_none_or(|at| at.elapsed() >= PROGRESS_INTERVAL)
        {
            tracking.last_progress = Some(Instant::now());
            self.notifications.status_changed();
        }
    }

    pub(crate) fn finished(
        &self,
        path: &Path,
        bytes: Option<u64>
    ) {
        let mut tracking = self.tracking.lock().unwrap();
        if let Some(resource) = tracking.resources.get(path) {
            let mut value = resource.lock().unwrap();
            value.active = false;
            value.complete = bytes.is_some();
            if let Some(bytes) = bytes {
                value.available = bytes;
                value.total = Some(bytes);
            }
        }
        self.notify_progress(&mut tracking);
    }

    pub(crate) fn cached(&self, tag: &str) {
        if let Some(attempt) = self.tracking.lock().unwrap().attempts.get_mut(tag) {
            attempt.stage = PreparationStage::Cached;
            attempt.failure = None;
        }
        self.notifications.status_changed();
    }

    pub(crate) fn failed(
        &self,
        tag: &str,
        stage: FailureStage
    ) {
        self.begin(tag);
        self.tracking
            .lock()
            .unwrap()
            .attempts
            .get_mut(tag)
            .unwrap()
            .failure = Some(PreparationFailure::new(stage));
        self.notifications.status_changed();
    }

    pub(crate) fn failure(&self, tag: &str) -> Option<PreparationFailure> {
        self.tracking
            .lock()
            .unwrap()
            .attempts
            .get(tag)?
            .failure
            .clone()
    }

    /// Content identity of the compiled predictor, for matching cached reports
    /// to the implementation the plane resolved (including after recompiles).
    pub(crate) fn build(&self, tag: &str) -> Option<String> {
        let tracking = self.tracking.lock().unwrap();
        let manifest = tracking.attempts.get(tag)?.manifest.as_ref()?;
        manifest
            .iter()
            .rev()
            .find(|item| item.kind == "dso")
            .map(|item| item.id.clone())
    }

    pub(crate) fn snapshot(&self, tag: &str) -> Option<PreparationProgress> {
        let tracking = self.tracking.lock().unwrap();
        let attempt = tracking.attempts.get(tag)?;
        let mut snapshot = PreparationProgress {
            stage: attempt.stage,
            resources_completed: 0,
            resources_total: attempt.manifest.as_ref().map(|items| items.len() as u64),
            bytes_available: 0,
            bytes_total: attempt.manifest.as_ref().map(|_| 0u64),
            active_resources: Vec::new(),
        };
        for item in attempt.manifest.iter().flatten() {
            let value = item.resource.lock().unwrap();
            snapshot.resources_completed += u64::from(value.complete);
            snapshot.bytes_available = snapshot.bytes_available.saturating_add(value.available);
            snapshot.bytes_total = snapshot
                .bytes_total
                .zip(value.total)
                .and_then(|(a, b)| a.checked_add(b));
            if value.active && snapshot.active_resources.len() < MAX_ACTIVE_RESOURCES {
                snapshot.active_resources.push(ActiveResource {
                    id: item.id.clone(),
                    name: item.name.clone(),
                    bytes_available: value.available,
                    bytes_total: value.total,
                });
            }
        }
        Some(snapshot)
    }

    pub(crate) fn forget(&self, tag: &str) {
        let mut tracking = self.tracking.lock().unwrap();
        tracking.attempts.remove(tag);
        // Cached files remain on disk; observations without subscribers need
        // not stay in RAM. In-flight callbacks safely recreate their entry.
        let used: HashSet<PathBuf> = tracking
            .attempts
            .values()
            .flat_map(|a| a.manifest.iter().flatten().map(|r| r.path.clone()))
            .collect();
        tracking.resources.retain(|path, _| used.contains(path));
    }
}

// Mirrors muna-rs PredictionService::get_resource_path. Signed query strings
// never enter the identity or the heartbeat payload.
fn resource_path(root: &Path, resource: &PredictionResource) -> (String, PathBuf) {
    let url = reqwest::Url::parse(&resource.url).ok();
    let stem = url
        .as_ref()
        .and_then(|u| u.path_segments())
        .and_then(|mut s| s.next_back())
        .unwrap_or("resource");
    let mut path = root.join(stem);
    if let Some(name) = &resource.name {
        path = path.join(name);
    }
    (stem.into(), path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> PreparationTracker {
        PreparationTracker::new(Arc::new(NotificationCenter::default()))
    }

    fn resource(id: &str, size: Option<u64>) -> PredictionResource {
        PredictionResource {
            kind: "dso".into(),
            url: format!("https://cdn.example/resources/{id}?signature=private"),
            name: None,
            size,
        }
    }

    fn root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("muna-preparation-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[tokio::test]
    async fn shared_resource_updates_both_manifests_without_combining_denominators() {
        let tracker = tracker();
        let root = root("shared");
        tracker
            .manifest(
                "a",
                &root,
                &[resource("shared", Some(100)), resource("a", Some(100))],
            )
            .await;
        tracker
            .manifest(
                "b",
                &root,
                &[resource("shared", Some(100)), resource("b", Some(300))],
            )
            .await;
        tracker.started(&root.join("shared"));
        tracker.advance(&root.join("shared"), 40, Some(100));
        assert_eq!(tracker.snapshot("a").unwrap().percent(), Some(20));
        assert_eq!(tracker.snapshot("b").unwrap().percent(), Some(10));
        tracker.finished(&root.join("shared"), Some(100));
        let a = tracker.snapshot("a").unwrap();
        assert_eq!(a.resources_completed, 1);
        assert_eq!(a.bytes_total, Some(200));
        assert_eq!(tracker.snapshot("b").unwrap().bytes_total, Some(400));
        let json = serde_json::to_string(&a).unwrap();
        assert!(!json.contains("signature"));
        assert!(!json.contains("https://"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn cache_hits_and_duplicate_manifest_entries_count_once() {
        let tracker = tracker();
        let root = root("cached");
        std::fs::write(root.join("cached"), [0u8; 25]).unwrap();
        tracker
            .manifest(
                "a",
                &root,
                &[resource("cached", None), resource("cached", Some(25))],
            )
            .await;
        let snapshot = tracker.snapshot("a").unwrap();
        assert_eq!(tracker.build("a").as_deref(), Some("cached"));
        assert_eq!(snapshot.resources_total, Some(1));
        assert_eq!(snapshot.resources_completed, 1);
        assert_eq!(snapshot.bytes_available, 25);
        assert_eq!(snapshot.bytes_total, Some(25));
        assert_eq!(snapshot.stage, PreparationStage::Cached);
        assert_eq!(snapshot.percent(), Some(100));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn unknown_sizes_stay_indeterminate_and_active_resources_are_bounded() {
        let tracker = tracker();
        let root = root("unknown");
        let resources: Vec<_> = (0..10).map(|i| resource(&i.to_string(), None)).collect();
        tracker.manifest("a", &root, &resources).await;
        for i in 0..10 {
            tracker.started(&root.join(i.to_string()));
        }
        let snapshot = tracker.snapshot("a").unwrap();
        assert_eq!(snapshot.bytes_total, None);
        assert_eq!(snapshot.percent(), None);
        assert_eq!(snapshot.active_resources.len(), MAX_ACTIVE_RESOURCES);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn retry_resets_partial_bytes_and_does_not_claim_completion_before_publish() {
        let tracker = tracker();
        let root = root("retry");
        tracker
            .manifest("a", &root, &[resource("x", Some(100))])
            .await;
        tracker.started(&root.join("x"));
        tracker.advance(&root.join("x"), 100, Some(100));
        assert_eq!(tracker.snapshot("a").unwrap().resources_completed, 0);
        assert_eq!(tracker.snapshot("a").unwrap().percent(), Some(99));
        tracker.finished(&root.join("x"), None);
        tracker.failed("a", FailureStage::Download);
        assert_eq!(
            tracker.failure("a").unwrap().code,
            "resource_download_failed"
        );
        tracker
            .manifest("a", &root, &[resource("x", Some(100))])
            .await;
        tracker.started(&root.join("x"));
        tracker.advance(&root.join("x"), 10, None);
        assert_eq!(tracker.snapshot("a").unwrap().bytes_available, 10);
        assert!(tracker.failure("a").is_none());
        tracker.forget("a");
        assert!(tracker.snapshot("a").is_none());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn byte_notifications_are_throttled_but_terminal_changes_are_immediate() {
        let tracker = tracker();
        let root = root("notifications");
        tracker
            .manifest("a", &root, &[resource("x", Some(100))])
            .await;
        tracker.notifications.status.notified().await;
        tracker.started(&root.join("x"));
        tracker.advance(&root.join("x"), 1, None);
        tracker.notifications.status.notified().await;
        tracker.advance(&root.join("x"), 1, None);
        assert!(tokio::time::timeout(
            Duration::from_millis(10),
            tracker.notifications.status.notified()
        )
        .await
        .is_err());
        tracker.cached("a");
        tokio::time::timeout(
            Duration::from_millis(10),
            tracker.notifications.status.notified(),
        )
        .await
        .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }
}
