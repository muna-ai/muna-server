/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Per-model load-state machine.
//!
//! Large predictors (LLM engines) can tens of seconds to cold-start, so the
//! server tracks each model's lifecycle explicitly: `Loading` (single-flight
//! warmup in progress), `Ready` (signature + batch plan known), or `Failed`.
//! Requests arriving during `Loading` wait up to a hold threshold, then get
//! `503` with a `Retry-After` derived from load-time history.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Signature, Value};
use muna::Muna;

use crate::metrics;
use crate::serving::batch::BatchPlan;
use crate::serving::predict;
use crate::serving::stats::ModelStats;

/// How long a request waits on a `Loading` model before giving up with 503.
const HOLD_THRESHOLD: Duration = Duration::from_secs(10);

/// Retry-After fallback before any load has completed.
const DEFAULT_LOAD_SECS: u64 = 30;

/// A model whose engine is warm and whose signature is known.
pub(crate) struct ReadyModel {
    pub loaded_at: Instant,
    /// Kept for future handler validation (the plan is derived from it now).
    #[allow(dead_code)]
    pub signature: Signature,
    pub plan: BatchPlan,
    pub stats: Arc<ModelStats>,
}

pub(crate) enum ModelState {
    Loading { since: Instant },
    Ready(Arc<ReadyModel>),
    Failed { error: String, at: Instant },
}

struct Slot {
    state: ModelState,
    watch: tokio::sync::watch::Sender<()>,
    /// An unload arrived while the load was in flight. The load cannot be
    /// cancelled (the sentinel prediction is executing inside native FFI on
    /// a blocking thread), so the load task performs the delete on
    /// completion instead -- nothing stays warm untracked.
    unload_requested: bool,
}

/// Error surface for `ensure_ready`.
#[derive(Debug)]
pub(crate) enum RegistryError {
    /// Model is still loading; retry after the given number of seconds.
    Loading { retry_after: u64 },
    /// The warmup prediction failed.
    Failed(String),
}

#[derive(Clone)]
pub(crate) struct ModelRegistry {
    inner: Arc<RegistryInner>,
}

/// Loader delegate: performs the warmup + signature fetch for one tag.
/// Injectable so single-flight behavior is testable without a live engine.
type Loader = Arc<
    dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<Signature, String>>
        + Send
        + Sync
>;

struct RegistryInner {
    muna: Arc<Muna>,
    models: DashMap<String, Slot>,
    loader: Loader,
    /// Most recent successful load duration in seconds, for Retry-After.
    last_load_secs: AtomicU64,
}

impl ModelRegistry {

    /// Create a model registry.
    pub(crate) fn new(muna: Arc<Muna>) -> Self {
        let loader_muna = muna.clone();
        let loader: Loader = Arc::new(move |tag| {
            let muna = loader_muna.clone();
            Box::pin(async move { load_model(&muna, &tag).await })
        });
        Self::with_loader(muna, loader)
    }

    fn with_loader(muna: Arc<Muna>, loader: Loader) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                muna,
                models: DashMap::new(),
                loader,
                last_load_secs: AtomicU64::new(DEFAULT_LOAD_SECS)
            })
        }
    }

    /// Get the model's `Ready` handle, warming it if necessary.
    ///
    /// Single-flight: the first request for a tag transitions the slot to
    /// `Loading` and spawns one warmup; concurrent requests await the watch
    /// channel. `Failed` is returned once and cleared so a later request may
    /// retry the load.
    pub(crate) async fn ensure_ready(
        &self,
        tag: &str
    ) -> Result<Arc<ReadyModel>, RegistryError> {
        let deadline = Instant::now() + HOLD_THRESHOLD;
        loop {
            // Map access is synchronous; waiting happens outside the shard lock.
            let mut waiter = match self.inner.models.entry(tag.to_string()) {
                dashmap::Entry::Occupied(entry) => match &entry.get().state {
                    ModelState::Ready(model) => return Ok(model.clone()),
                    ModelState::Failed { error, .. } => {
                        let error = error.clone();
                        entry.remove();
                        return Err(RegistryError::Failed(error));
                    }
                    ModelState::Loading { since } => {
                        let since = *since;
                        if Instant::now() >= deadline {
                            return Err(RegistryError::Loading {
                                retry_after: self.retry_after(since)
                            });
                        }
                        entry.get().watch.subscribe()
                    }
                },
                dashmap::Entry::Vacant(entry) => {
                    let (watch, rx) = tokio::sync::watch::channel(());
                    entry.insert(Slot {
                        state: ModelState::Loading { since: Instant::now() },
                        watch,
                        unload_requested: false
                    });
                    self.spawn_load(tag.to_string());
                    rx
                }
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if tokio::time::timeout(remaining, waiter.changed()).await.is_err() {
                let since = self.inner.models.get(tag).and_then(|slot| match &slot.state {
                    ModelState::Loading { since } => Some(*since),
                    _ => None,
                });
                match since {
                    Some(since) => {
                        return Err(RegistryError::Loading {
                            retry_after: self.retry_after(since)
                        });
                    }
                    // State changed while timing out; loop re-reads it.
                    None => continue,
                }
            }
        }
    }

    /// Warm a model without waiting for it (heartbeat reconciliation path).
    /// Idempotent: a `Ready` or `Loading` tag is a no-op; a stale failure is
    /// cleared so reconciliation can retry.
    pub(crate) fn warm(&self, tag: &str) {
        match self.inner.models.entry(tag.to_string()) {
            dashmap::Entry::Occupied(entry) => {
                if matches!(entry.get().state, ModelState::Failed { .. }) {
                    entry.remove();
                    self.warm(tag);
                }
            }
            dashmap::Entry::Vacant(entry) => {
                let (watch, _rx) = tokio::sync::watch::channel(());
                entry.insert(Slot {
                    state: ModelState::Loading { since: Instant::now() },
                    watch,
                    unload_requested: false
                });
                self.spawn_load(tag.to_string());
            }
        }
    }

    /// Unload a model. Idempotent: an absent tag (or a repeated unload of a
    /// still-loading tag) is a no-op. Unloading a `Loading` tag defers the
    /// delete to the load task (see `Slot::unload_requested`).
    pub(crate) async fn unload(&self, tag: &str) {
        if let Some(mut slot) = self.inner.models.get_mut(tag) {
            if matches!(slot.state, ModelState::Loading { .. }) {
                slot.unload_requested = true;
                tracing::info!(tag = %tag, "unload deferred until in-flight load completes");
                return;
            }
        }
        if self.inner.models.remove(tag).is_none() {
            return;
        }
        delete_predictor(&self.inner.muna, tag).await;
    }

    /// Snapshot every slot's state for status reporting.
    pub(crate) fn snapshot(&self) -> Vec<(String, ModelState)> {
        self.inner.models.iter()
            .map(|entry| {
                let state = match &entry.state {
                    ModelState::Loading { since } => ModelState::Loading { since: *since },
                    ModelState::Ready(model) => ModelState::Ready(model.clone()),
                    ModelState::Failed { error, at } => ModelState::Failed {
                        error: error.clone(),
                        at: *at
                    },
                };
                (entry.key().clone(), state)
            })
            .collect()
    }

    /// Tags currently in the `Ready` state.
    pub(crate) fn ready_tags(&self) -> Vec<String> {
        self.inner.models.iter()
            .filter(|entry| matches!(entry.state, ModelState::Ready(_)))
            .map(|entry| entry.key().clone())
            .collect()
    }

    fn retry_after(&self, since: Instant) -> u64 {
        let expected = self.inner.last_load_secs.load(Ordering::Relaxed);
        expected.saturating_sub(since.elapsed().as_secs()).max(5)
    }

    /// Spawn the single-flight warmup: sentinel prediction (reaches the
    /// engine's `load_predictor`), then signature fetch and plan derivation.
    fn spawn_load(&self, tag: String) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            let vram_before = metrics::total_memory_used_mb();
            let stats = Arc::new(ModelStats::new());
            let outcome = (inner.loader)(tag.clone()).await;
            let vram_after = metrics::total_memory_used_mb();
            if let (Some(before), Some(after)) = (vram_before, vram_after) {
                if after > before {
                    stats.record_vram(after - before);
                }
            }
            stats.record_load_time(start.elapsed());
            let state = match outcome {
                Ok(signature) => {
                    inner.last_load_secs.store(
                        start.elapsed().as_secs().max(1),
                        Ordering::Relaxed
                    );
                    tracing::info!(
                        tag = %tag,
                        load_time_ms = %format!("{:.0}", start.elapsed().as_secs_f64() * 1000.0),
                        "model ready"
                    );
                    ModelState::Ready(Arc::new(ReadyModel {
                        loaded_at: Instant::now(),
                        plan: BatchPlan::from_signature(&signature),
                        signature,
                        stats
                    }))
                }
                Err(error) => {
                    tracing::error!(tag = %tag, error = %error, "model load failed");
                    ModelState::Failed { error, at: Instant::now() }
                }
            };
            let mut delete_after_load = false;
            if let Some(mut slot) = inner.models.get_mut(&tag) {
                if slot.unload_requested {
                    delete_after_load = true;
                } else {
                    slot.state = state;
                    let _ = slot.watch.send(());
                }
            }
            if delete_after_load {
                // Removing the slot drops the watch sender; waiters re-read
                // the map, find it vacant, and re-warm organically if they
                // still want the model.
                inner.models.remove(&tag);
                delete_predictor(&inner.muna, &tag).await;
            }
        });
    }
}

async fn delete_predictor(muna: &Arc<Muna>, tag: &str) {
    let delete_muna = muna.clone();
    let delete_tag = tag.to_string();
    let result = predict::run(move || async move {
        delete_muna.predictions.delete(&delete_tag).await
    }).await;
    match result {
        Ok(_) => tracing::info!(tag = %tag, "model unloaded"),
        Err(e) => tracing::warn!(tag = %tag, error = %e, "failed to unload predictor"),
    }
}

async fn load_model(
    muna: &Arc<Muna>,
    tag: &str
) -> Result<Signature, String> {
    // Preload convention: create a prediction that deliberately excludes
    // the predictor's required inputs. Loading the predictor runs all
    // constructors and initializers (the actual engine load); the
    // prediction itself then exits early on the missing argument, so
    // `prediction.error` is expected and deliberately ignored. Genuine
    // load failures (download, native predictor creation) surface as an
    // `Err` from `create` itself.
    let warm_muna = muna.clone();
    let warm_tag = tag.to_string();
    predict::run(move || async move {
        let inputs = HashMap::from([("_".to_string(), Value::Null)]);
        warm_muna.predictions.create(
            &warm_tag,
            Some(inputs),
            Some(Acceleration::LocalGpu),
            None,
            None
        ).await
    }).await.map_err(|e| e.to_string())?;
    let sig_muna = muna.clone();
    let sig_tag = tag.to_string();
    let predictor = predict::run(move || async move {
        sig_muna.predictors.retrieve(&sig_tag).await
    }).await.map_err(|e| e.to_string())?;
    let predictor = predictor.ok_or_else(|| format!("predictor {tag} not found"))?;
    Ok(predictor.signature)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    fn registry_with(
        loads: Arc<AtomicUsize>,
        fail: bool,
        delay: Duration
    ) -> ModelRegistry {
        let loader: Loader = Arc::new(move |_tag| {
            let loads = loads.clone();
            Box::pin(async move {
                loads.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                if fail {
                    Err("boom".to_string())
                } else {
                    Ok(Signature { inputs: vec![], outputs: vec![] })
                }
            })
        });
        ModelRegistry::with_loader(Arc::new(Muna::new(None, None)), loader)
    }

    #[tokio::test]
    async fn single_flight_one_warmup_for_concurrent_requests() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), false, Duration::from_millis(100));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let registry = registry.clone();
            handles.push(tokio::spawn(async move {
                registry.ensure_ready("@test/model").await
            }));
        }
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_load_returned_once_then_retried() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), true, Duration::from_millis(10));
        let first = registry.ensure_ready("@test/model").await;
        assert!(matches!(first, Err(RegistryError::Failed(_))));
        // The failure was cleared; a second request retries the load.
        let second = registry.ensure_ready("@test/model").await;
        assert!(matches!(second, Err(RegistryError::Failed(_))));
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn warm_is_idempotent_while_loading() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), false, Duration::from_millis(100));
        registry.warm("@test/model");
        registry.warm("@test/model");
        registry.warm("@test/model");
        let model = registry.ensure_ready("@test/model").await;
        assert!(model.is_ok());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unload_during_load_discards_model() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), false, Duration::from_millis(150));
        registry.warm("@test/model");
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Unload while the load is in flight: the delete is deferred to the
        // load task, and the model must not surface as Ready afterwards.
        registry.unload("@test/model").await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(registry.ready_tags().is_empty());
        assert!(registry.snapshot().is_empty());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ready_tags_reports_only_ready() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), false, Duration::from_millis(10));
        assert!(registry.ready_tags().is_empty());
        registry.ensure_ready("@test/model").await.unwrap();
        assert_eq!(registry.ready_tags(), vec!["@test/model".to_string()]);
    }
}
