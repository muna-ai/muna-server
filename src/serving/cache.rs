/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Recorded cached-tier state: which predictors' resources are known to be
//! downloaded on this node.
//!
//! Cached-ness is RECORDED state, not probed state: an entry is written
//! whenever this server itself downloads a predictor's resources (a prefetch
//! directive, an engine load, or the `preload` CLI at image-bake time). The
//! server never walks the muna client's cache directory and never deletes
//! resources. If disk contents vanish behind the server's back, the next
//! load re-downloads through the normal client path and re-records --
//! a mis-reported "cached" costs one slow load, never a wrong result.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use muna::types::{Acceleration, Prediction, Value};
use muna::Muna;
use serde::{Deserialize, Serialize};

use crate::serving::predict;

/// One recorded download outcome.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct ManifestEntry {
    /// Resource names (or URL basenames) from the recording prediction.
    pub resources: Vec<String>,
    /// Unix timestamp of the record.
    pub created: u64,
}

/// Persistent map of predictor tag -> recorded download outcome, backed by
/// one `predictors.json` on the same volume as the muna resource cache it
/// describes (manifest and resources share fate).
///
/// Cheap to clone (shared inner). Single writer by construction: only this
/// process records, and every record rewrites the file atomically (temp
/// file + rename), so a mid-write crash can at worst lose the newest entry.
#[derive(Clone)]
pub(crate) struct ManifestStore {
    inner: Arc<StoreInner>,
}

struct StoreInner {
    path: PathBuf,
    entries: Mutex<BTreeMap<String, ManifestEntry>>,
    /// Tags with a prefetch download in flight (single-flight guard).
    inflight: DashMap<String, ()>,
}

impl ManifestStore {

    /// Open the store, seeding from `path` if it exists. A malformed file is
    /// logged and treated as empty (self-heals as downloads re-record).
    pub(crate) fn open(path: PathBuf) -> Self {
        let entries = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(entries) => entries,
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "malformed predictor manifest; starting empty"
                    );
                    BTreeMap::new()
                }
            },
            Err(_) => BTreeMap::new(),
        };
        Self {
            inner: Arc::new(StoreInner {
                path,
                entries: Mutex::new(entries),
                inflight: DashMap::new()
            })
        }
    }

    /// Record a successful download from the prediction that performed it.
    /// Upserts: re-recording a tag (e.g. after a re-push) rewrites its entry.
    ///
    /// Note the prediction's resource URLs are the upstream URLs as-is (the
    /// client does not rewrite them to local paths); they are recorded for
    /// bookkeeping only.
    pub(crate) fn record(&self, tag: &str, prediction: &Prediction) {
        let resources = prediction.resources.as_deref().unwrap_or_default();
        let names = resources.iter()
            .map(|r| {
                r.name.clone().unwrap_or_else(|| {
                    r.url.rsplit('/').next().unwrap_or(&r.url).to_string()
                })
            })
            .collect();
        let entry = ManifestEntry {
            resources: names,
            created: unix_now()
        };
        let snapshot = {
            let mut entries = self.inner.entries.lock().unwrap();
            entries.insert(tag.to_string(), entry);
            entries.clone()
        };
        self.persist(&snapshot);
    }

    /// Whether a tag has a recorded download.
    pub(crate) fn contains(&self, tag: &str) -> bool {
        self.inner.entries.lock().unwrap().contains_key(tag)
    }

    /// Path of the backing `predictors.json` (the data volume anchor).
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.inner.path
    }

    /// Every recorded tag, for status reporting.
    pub(crate) fn cached(&self) -> Vec<String> {
        self.inner.entries.lock().unwrap()
            .keys()
            .cloned()
            .collect()
    }

    /// Ensure a tag's resources are downloaded (the cached tier), in the
    /// background. Idempotent and single-flight: already-recorded tags and
    /// tags with a download in flight are no-ops. Failures are logged; the
    /// tag's absence from the next heartbeat is the report.
    pub(crate) fn prefetch(&self, muna: Arc<Muna>, tag: &str) {
        if self.contains(tag) {
            return;
        }
        if self.inner.inflight.insert(tag.to_string(), ()).is_some() {
            return;
        }
        let store = self.clone();
        let tag = tag.to_string();
        tokio::spawn(async move {
            let download_muna = muna.clone();
            let download_tag = tag.clone();
            // Download-only convention: empty (not absent) inputs fetch the
            // predictor's resources without loading the engine. Same shape
            // as the `preload` CLI subcommand.
            let result = predict::run(move || async move {
                download_muna.predictions.create(
                    &download_tag,
                    Some(HashMap::<String, Value>::new()),
                    Some(Acceleration::LocalGpu),
                    None,
                    None
                ).await
            }).await;
            match result {
                Ok(prediction) => {
                    store.record(&tag, &prediction);
                    tracing::info!(tag = %tag, "prefetched predictor resources");
                }
                Err(e) => {
                    tracing::warn!(tag = %tag, error = %e, "prefetch failed");
                }
            }
            store.inner.inflight.remove(&tag);
        });
    }

    /// Atomically rewrite `predictors.json` (write temp + rename).
    fn persist(&self, entries: &BTreeMap<String, ManifestEntry>) {
        let path = &self.inner.path;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json = match serde_json::to_vec_pretty(entries) {
            Ok(json) => json,
            Err(e) => {
                tracing::error!(error = %e, "failed to serialize predictor manifest");
                return;
            }
        };
        let temp = path.with_extension("json.tmp");
        let result = std::fs::write(&temp, json)
            .and_then(|_| std::fs::rename(&temp, path));
        if let Err(e) = result {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to persist predictor manifest"
            );
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use muna::types::PredictionResource;

    use super::*;

    static UNIQUE: AtomicU64 = AtomicU64::new(0);

    fn temp_path() -> PathBuf {
        std::env::temp_dir()
            .join(format!(
                "muna-server-cache-test-{}-{}",
                std::process::id(),
                UNIQUE.fetch_add(1, Ordering::Relaxed)
            ))
            .join("predictors.json")
    }

    fn prediction_with_resources(resources: Vec<PredictionResource>) -> Prediction {
        Prediction {
            id: "pred_test".into(),
            tag: "@test/model".into(),
            created: "0".into(),
            configuration: None,
            resources: Some(resources),
            results: None,
            latency: None,
            error: None,
            logs: None,
        }
    }

    #[test]
    fn record_then_reopen_roundtrip() {
        let path = temp_path();
        let store = ManifestStore::open(path.clone());
        let prediction = prediction_with_resources(vec![PredictionResource {
            kind: "dso".into(),
            url: "https://cdn.example/resources/weights.bin".into(),
            name: Some("weights.bin".into()),
        }]);
        store.record("@test/model", &prediction);
        assert!(store.contains("@test/model"));
        // A fresh store seeded from the same file sees the record.
        let reopened = ManifestStore::open(path);
        assert!(reopened.contains("@test/model"));
        assert_eq!(reopened.cached(), vec!["@test/model".to_string()]);
    }

    #[test]
    fn rerecord_overwrites_entry() {
        let store = ManifestStore::open(temp_path());
        let first = prediction_with_resources(vec![PredictionResource {
            kind: "dso".into(),
            url: "https://cdn.example/a.bin".into(),
            name: Some("a.bin".into()),
        }]);
        let second = prediction_with_resources(vec![PredictionResource {
            kind: "dso".into(),
            url: "https://cdn.example/b.bin".into(),
            name: Some("b.bin".into()),
        }]);
        store.record("@test/model", &first);
        store.record("@test/model", &second);
        assert_eq!(store.cached().len(), 1);
        let entries = store.inner.entries.lock().unwrap();
        assert_eq!(entries["@test/model"].resources, vec!["b.bin"]);
    }

    #[test]
    fn open_missing_file_starts_empty() {
        let store = ManifestStore::open(temp_path());
        assert!(store.cached().is_empty());
        assert!(!store.contains("@test/model"));
    }

    #[test]
    fn open_malformed_file_starts_empty() {
        let path = temp_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json").unwrap();
        let store = ManifestStore::open(path);
        assert!(store.cached().is_empty());
    }
}
