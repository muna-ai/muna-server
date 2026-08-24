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

/// Tracks which tags have complete resources on disk. Cloneable handle;
/// state is shared.
#[derive(Clone)]
pub(crate) struct CacheTracker {
    /// Per-tag deployment keys for building keyed download clients.
    keys: KeyStore,
    states: Arc<DashMap<String, CacheState>>,
}

impl CacheTracker {

    pub(crate) fn new(keys: KeyStore) -> Self {
        Self { keys, states: Arc::new(DashMap::new()) }
    }

    /// Ensure the tag's resources are on disk. Idempotent and single-flight:
    /// a tag already caching or cached is a no-op; a failed tag retries
    /// only after backoff.
    pub(crate) fn ensure_cached(&self, tag: &str) {
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
        self.spawn_download(tag.to_string());
    }

    /// Drop a failed cache record (nothing was achieved on disk, so there
    /// is nothing truthful to keep reporting). `Cached` records are kept:
    /// the resources genuinely are on disk, and the plane's placement wants
    /// to see that even under a `none` goal.
    pub(crate) fn forget_failed(&self, tag: &str) {
        self.states
            .remove_if(tag, |_, state| matches!(state, CacheState::Failed { .. }));
    }

    /// Snapshot every tracked tag for status reporting.
    pub(crate) fn snapshot(&self) -> Vec<(String, CacheState)> {
        self.states
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    fn spawn_download(&self, tag: String) {
        // Ephemeral keyed instance per download: the download-only
        // prediction localizes resources without loading a native
        // predictor, so no handle outlives this call (unlike the
        // registry's persistent per-model instance).
        let key = self.keys.get(&tag).map(|entry| entry.value().clone());
        let muna = Arc::new(Muna::with_client(Arc::new(ServerClient::with_key(key))));
        let states = self.states.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            let download_muna = muna.clone();
            let download_tag = tag.clone();
            // Download-only convention: an empty (but present) inputs map
            // makes the muna client create a raw prediction and localize
            // its resources without loading any engine. Acceleration must
            // be a LOCAL flavor: without it the API resolves the tag as a
            // remote predictor, which compiled models do not have.
            let result = predict::run(move || async move {
                download_muna.predictions.create(
                    &download_tag,
                    Some(HashMap::<String, Value>::new()),
                    Some(Acceleration::LocalAuto),
                    None,
                    None
                ).await
            }).await;
            let state = match result {
                Ok(prediction) => match prediction.error {
                    Some(error) => {
                        tracing::warn!(tag = %tag, error = %error, "cache download failed");
                        CacheState::Failed { error, at: Instant::now() }
                    }
                    None => {
                        tracing::info!(
                            tag = %tag,
                            elapsed_ms = %format!("{:.0}", start.elapsed().as_secs_f64() * 1000.0),
                            "model cached on disk"
                        );
                        CacheState::Cached
                    }
                },
                Err(e) => {
                    tracing::warn!(tag = %tag, error = %e, "cache download failed");
                    CacheState::Failed { error: e.to_string(), at: Instant::now() }
                }
            };
            states.insert(tag, state);
        });
    }
}
