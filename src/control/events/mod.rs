/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Completed loads are facts, independent of the latest residency snapshot.
//! Keep them on disk until the control plane acknowledges a database commit.

mod storage;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::notifications::NotificationCenter;

/// Mirrored in control-plane/src/protocol/load_events.rs.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ModelLoadEvent {
    pub id: String,
    pub tag: String,
    pub created: DateTime<Utc>,
    pub duration_ms: f64,
    pub download_ms: f64,
    pub server_version: String,
    pub device_ids: Option<Vec<u32>>,
    pub build: Option<String>,
    pub vram_mb: Option<u64>,
    pub gpu_family: Option<String>,
    pub gpu_name: Option<String>,
    pub concurrent_loads: u32,
}

#[derive(Clone)]
struct Pending {
    event: ModelLoadEvent,
    durable: bool,
    acknowledged: bool,
}

pub(crate) struct LoadEvents {
    path: PathBuf,
    pending: Mutex<BTreeMap<String, Pending>>,
    changed: Notify,
    notifications: Arc<NotificationCenter>,
}

impl LoadEvents {
    /// Namespace by node identity: a reused cache must not attribute another
    /// node's historical loads to its replacement.
    pub(crate) fn open(cache: &Path, node_id: &str, notifications: Arc<NotificationCenter>) -> std::io::Result<Self> {
        use sha2::{Digest, Sha256};
        let path = cache.join("load-events").join(format!("{:x}", Sha256::digest(node_id.as_bytes())));
        let events = storage::read(&path)?;
        let pending = events.into_iter().map(|event| {
            (event.id.clone(), Pending { event, durable: true, acknowledged: false })
        }).collect();
        Ok(Self { path, pending: Mutex::new(pending), changed: Notify::new(), notifications })
    }

    /// Called exactly once on successful load completion, before publishing
    /// Ready. Local persistence runs on a blocking worker, outside inference.
    pub(crate) fn record(&self, event: ModelLoadEvent) {
        self.pending.lock().unwrap().insert(event.id.clone(), Pending {
            event, durable: false, acknowledged: false,
        });
        self.changed.notify_one();
    }

    /// Bounded wire batches; unacknowledged events are never dropped.
    pub(crate) fn batch(&self) -> Vec<ModelLoadEvent> {
        self.pending.lock().unwrap().values()
            .filter(|pending| pending.durable && !pending.acknowledged)
            .take(64).map(|pending| pending.event.clone()).collect()
    }

    pub(crate) fn acknowledge(&self, ids: &[String]) {
        let mut pending = self.pending.lock().unwrap();
        for id in ids {
            if let Some(entry) = pending.get_mut(id).filter(|entry| entry.durable) {
                entry.acknowledged = true;
            }
        }
        drop(pending);
        self.changed.notify_one();
        // Drain the next batch without waiting for the periodic heartbeat.
        if !ids.is_empty() && !self.batch().is_empty() {
            self.notifications.status_changed();
        }
    }

    fn flush(&self) -> std::io::Result<()> {
        let pending: Vec<_> = self.pending.lock().unwrap().values()
            .filter(|entry| !entry.durable || entry.acknowledged).cloned().collect();
        let mut published = false;
        for entry in pending {
            if entry.acknowledged {
                storage::remove(&self.path, &entry.event.id)?;
                self.pending.lock().unwrap().remove(&entry.event.id);
            } else {
                storage::write(&self.path, &entry.event)?;
                if let Some(entry) = self.pending.lock().unwrap().get_mut(&entry.event.id) {
                    entry.durable = true;
                }
                published = true;
            }
        }
        if published {
            self.notifications.status_changed();
        }
        Ok(())
    }

    pub(crate) async fn run(self: Arc<Self>, shutdown: &CancellationToken) {
        let mut retry = tokio::time::interval(Duration::from_secs(1));
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {},
                _ = self.changed.notified() => {},
                _ = retry.tick() => {},
            }
            let events = self.clone();
            match tokio::task::spawn_blocking(move || events.flush()).await {
                Ok(Ok(())) => {},
                result => tracing::error!(?result, "load event outbox persistence failed; retaining events for retry"),
            }
            if shutdown.is_cancelled() { break; }
        }
    }
}

#[cfg(test)]
mod tests;
