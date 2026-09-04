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
//! `429` with a `Retry-After` derived from load-time history.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Signature, Value};
use muna::Muna;

use crate::client::ServerClient;
use crate::metrics;
use crate::serving::batch::BatchPlan;
use crate::serving::predict;
use crate::serving::stats::ModelStats;
use crate::state::KeyStore;

/// How long a request waits on a `Loading` model before giving up with 429.
const HOLD_THRESHOLD: Duration = Duration::from_secs(10);

/// Retry-After fallback before any load has completed.
const DEFAULT_LOAD_SECS: u64 = 30;

/// Reconciliation-path retry backoff for failed loads. Under a declarative
/// goal the control plane re-asserts `process` every beat; without this a node
/// that just failed would hot-loop engine loads at heartbeat cadence.
const FAILED_RETRY_BACKOFF: Duration = Duration::from_secs(60);

/// A model whose engine is warm and whose signature is known.
pub(crate) struct ReadyModel {
    /// The per-model Muna client that performed the warm load. It OWNS the
    /// loaded native predictor handle (muna-rs caches handles per client
    /// instance), so it must live exactly as long as the model: every
    /// prediction goes through it, and unload deletes through it -- a
    /// delete through any other instance would silently leak the handle.
    /// Also carries the model's deployment key when the control plane
    /// supplied one.
    pub muna: Arc<Muna>,
    /// When the load completed (reported as the model's `created` age by
    /// the OpenAI-compatible `/v1/models` handler).
    pub loaded_at: Instant,
    /// Kept for future handler validation (the plan is derived from it now).
    #[allow(dead_code)]
    pub signature: Signature,
    /// How the dispatcher batches predictions for this model, derived from
    /// the signature's batch config.
    pub plan: BatchPlan,
    /// Live counters (queue depth, prediction totals, latency, load time,
    /// VRAM), shared with the dispatcher and reported in heartbeats.
    pub stats: Arc<ModelStats>,
}

/// Lifecycle state of one model. Absent-from-map is the implicit fourth
/// state (never requested, or unloaded); `Failed` is cleared on read so the
/// next request retries the load.
pub(crate) enum ModelState {
    /// Single-flight warmup in progress since the given instant (used to
    /// derive `Retry-After` for requests that give up waiting).
    Loading { since: Instant },
    /// Engine warm, signature and batch plan known; shared with every
    /// request currently predicting against the model.
    Ready(Arc<ReadyModel>),
    /// The warmup failed at the given instant; `error` is surfaced to the
    /// requester (and the control plane) verbatim.
    Failed { error: String, at: Instant },
}

/// One tag's entry in the registry map: the model's current lifecycle state
/// plus the coordination bits that make loads single-flight.
struct Slot {
    /// Current lifecycle state (`Loading` -> `Ready` | `Failed`).
    state: ModelState,
    /// Notifies requests waiting out a `Loading` state. The load task sends
    /// on completion; `ensure_ready` waiters subscribe and then re-read
    /// `state` from the map. Dropping the slot drops the sender, which also
    /// wakes waiters (they find the entry vacant and re-warm organically).
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
    /// The tag is not in this server's pinned model set (`--models`).
    NotServed(String),
}

#[derive(Clone)]
pub(crate) struct ModelRegistry {
    inner: Arc<RegistryInner>,
}

/// What a successful warmup yields: the signature, the per-model Muna
/// client that owns the loaded handle, and how long each phase took.
pub(crate) struct Loaded {
    /// Predictor signature.
    pub signature: Signature,
    /// Muna client.
    pub muna: Arc<Muna>,
    /// Resource download that preceded the load (near zero when
    /// disk-resident).
    pub download: Duration,
    /// Predictor creation with every resource on disk: the cold start.
    pub load: Duration,
}

/// Loader delegate: performs the warmup + signature fetch for one tag.
/// Injectable so single-flight behavior is testable without a live engine.
type Loader = Arc<
    dyn Fn(String) -> futures_util::future::BoxFuture<'static, Result<Loaded, String>>
        + Send
        + Sync
>;

/// Shared state behind the cloneable `ModelRegistry` handle.
struct RegistryInner {
    /// Per-tag slots; a tag's absence means it was never loaded (or was
    /// unloaded).
    models: DashMap<String, Slot>,
    /// Warmup delegate invoked by `spawn_load` (injectable for tests).
    loader: Loader,
    /// Pinned model set (`--models`): the only tags this server will load.
    /// `None` means open behavior (any tag loadable on demand).
    pinned: Option<HashSet<String>>,
    /// Most recent successful load duration in seconds, for Retry-After.
    last_load_secs: AtomicU64,
}

impl ModelRegistry {

    /// Create a model registry. `keys` holds the per-tag deployment keys
    /// delivered by control-plane residency directives; tags without one
    /// load through the process-wide `$MUNA_ACCESS_KEY`.
    pub(crate) fn new(
        keys: KeyStore,
        pinned: Option<HashSet<String>>
    ) -> Self {
        let loader: Loader = Arc::new(move |tag| {
            let key = keys.get(&tag).map(|entry| entry.value().clone());
            Box::pin(async move { load_model(&tag, key).await })
        });
        Self::with_loader(loader, pinned)
    }

    fn with_loader(
        loader: Loader,
        pinned: Option<HashSet<String>>
    ) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                models: DashMap::new(),
                loader,
                pinned,
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
        if !self.serves(tag) {
            return Err(RegistryError::NotServed(tag.to_string()));
        }
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

    /// Whether this server serves the tag (member of the pinned set, or no
    /// pinned set configured).
    pub(crate) fn serves(&self, tag: &str) -> bool {
        self.inner
            .pinned
            .as_ref()
            .is_none_or(|pinned| pinned.contains(tag))
    }

    /// Warm a model without waiting for it (heartbeat reconciliation path).
    /// Idempotent: a `Ready` or `Loading` tag is a no-op; a stale failure is
    /// cleared so reconciliation can retry.
    ///
    /// A warm directive for a tag outside the pinned set is a control-plane
    /// misconfiguration: logged and ignored.
    pub(crate) fn warm(&self, tag: &str) {
        if !self.serves(tag) {
            tracing::warn!(tag = %tag, "warm ignored: tag is not in this server's --models set");
            return;
        }
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

    /// Warm a model on the reconciliation path: like [`warm`](Self::warm),
    /// but a recent failure is left to cool down instead of retrying every
    /// heartbeat (failed stickiness -- the failure stays visible in status
    /// reports while the backoff runs, so the plane can place elsewhere).
    pub(crate) fn warm_reconcile(&self, tag: &str) {
        if let Some(slot) = self.inner.models.get(tag) {
            if let ModelState::Failed { at, .. } = &slot.state {
                if at.elapsed() < FAILED_RETRY_BACKOFF {
                    return;
                }
            }
        }
        self.warm(tag);
    }

    /// Unload a model. Idempotent: an absent tag (or a repeated unload of a
    /// still-loading tag) is a no-op. Unloading a `Loading` tag defers the
    /// delete to the load task (see `Slot::unload_requested`).
    ///
    /// Two-step teardown: (1) delete the predictor through the model's OWN
    /// Muna instance (the handle cache is per-instance, so no other client
    /// can release it), then (2) drop the slot so the per-model `Arc<Muna>`
    /// itself drops once in-flight predictions holding the `Arc<ReadyModel>`
    /// finish.
    pub(crate) async fn unload(&self, tag: &str) {
        if let Some(mut slot) = self.inner.models.get_mut(tag) {
            if matches!(slot.state, ModelState::Loading { .. }) {
                slot.unload_requested = true;
                tracing::info!(tag = %tag, "unload deferred until in-flight load completes");
                return;
            }
        }
        let Some((_, slot)) = self.inner.models.remove(tag) else {
            return;
        };
        if let ModelState::Ready(model) = slot.state {
            delete_predictor(&model.muna, tag).await;
        }
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

    /// The model's `Ready` handle, if currently loaded. Never triggers a
    /// load (unlike `ensure_ready`).
    pub(crate) fn ready(&self, tag: &str) -> Option<Arc<ReadyModel>> {
        self.inner.models.get(tag).and_then(|slot| match &slot.state {
            ModelState::Ready(model) => Some(model.clone()),
            _ => None,
        })
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
            // Kept out of `state` so a deferred unload can delete through
            // the instance that actually loaded the handle.
            let mut loaded_muna: Option<Arc<Muna>> = None;
            let state = match outcome {
                Ok(Loaded { signature, muna, download, load }) => {
                    stats.record_download_time(download);
                    stats.record_load_time(load);
                    // Retry-After covers the whole wait a caller sees,
                    // download included.
                    inner.last_load_secs.store(
                        start.elapsed().as_secs().max(1),
                        Ordering::Relaxed
                    );
                    tracing::info!(
                        tag = %tag,
                        download_time_ms = %format!("{:.0}", download.as_secs_f64() * 1000.0),
                        load_time_ms = %format!("{:.0}", load.as_secs_f64() * 1000.0),
                        "model ready"
                    );
                    loaded_muna = Some(muna.clone());
                    ModelState::Ready(Arc::new(ReadyModel {
                        muna,
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
                if let Some(muna) = loaded_muna {
                    delete_predictor(&muna, &tag).await;
                }
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
    tag: &str,
    key: Option<String>
) -> Result<Loaded, String> {
    // One Muna instance per model, keyed with the tag's deployment key when
    // the control plane supplied one. The instance persists on `ReadyModel`
    // because it owns the native predictor handle loaded below.
    let muna = Arc::new(Muna::with_client(Arc::new(ServerClient::with_key(key))));
    // Phase 1, download. The download-only convention (an empty but present
    // inputs map; see cache.rs) localizes every resource without loading an
    // engine. Doing it as its own step keeps the load timing below a pure
    // cold start: a node that has never seen the tag pays the fetch here,
    // a disk-resident node finds everything cached and passes through in
    // well under a second. Same acceleration as the load so the resource
    // set is identical.
    let download_started = Instant::now();
    let download_muna = muna.clone();
    let download_tag = tag.to_string();
    let downloaded = predict::run(move || async move {
        download_muna.predictions.create(
            &download_tag,
            Some(HashMap::<String, Value>::new()),
            Some(Acceleration::LocalGpu),
            None,
            None
        ).await
    }).await.map_err(|e| e.to_string())?;
    if let Some(error) = downloaded.error {
        return Err(error);
    }
    let download = download_started.elapsed();
    // Phase 2, load. Preload convention: create a prediction that
    // deliberately excludes the predictor's required inputs. Loading the
    // predictor runs all constructors and initializers (the actual engine
    // load); the prediction itself then exits early on the missing
    // argument, so `prediction.error` is expected and deliberately
    // ignored. Genuine load failures (native predictor creation) surface
    // as an `Err` from `create` itself.
    let load_started = Instant::now();
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
    let load = load_started.elapsed();
    let sig_muna = muna.clone();
    let sig_tag = tag.to_string();
    let predictor = predict::run(move || async move {
        sig_muna.predictors.retrieve(&sig_tag).await
    }).await.map_err(|e| e.to_string())?;
    let predictor = predictor.ok_or_else(|| format!("predictor {tag} not found"))?;
    Ok(Loaded { signature: predictor.signature, muna, download, load })
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
        registry_pinned(loads, fail, delay, None)
    }

    fn registry_pinned(
        loads: Arc<AtomicUsize>,
        fail: bool,
        delay: Duration,
        pinned: Option<HashSet<String>>
    ) -> ModelRegistry {
        let loader: Loader = Arc::new(move |_tag| {
            let loads = loads.clone();
            Box::pin(async move {
                loads.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(delay).await;
                if fail {
                    Err("boom".to_string())
                } else {
                    Ok(Loaded {
                        signature: Signature { inputs: vec![], outputs: vec![] },
                        muna: Arc::new(Muna::new(None, None)),
                        download: Duration::ZERO,
                        load: delay
                    })
                }
            })
        });
        ModelRegistry::with_loader(loader, pinned)
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
    async fn pinned_set_rejects_unlisted_tag() {
        let loads = Arc::new(AtomicUsize::new(0));
        let pinned = HashSet::from(["@test/served".to_string()]);
        let registry = registry_pinned(
            loads.clone(),
            false,
            Duration::from_millis(10),
            Some(pinned)
        );
        let other = registry.ensure_ready("@test/other").await;
        assert!(matches!(other, Err(RegistryError::NotServed(_))));
        // A warm directive for an unlisted tag is ignored, not loaded.
        registry.warm("@test/other");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(loads.load(Ordering::SeqCst), 0);
        // The pinned member loads normally.
        let served = registry.ensure_ready("@test/served").await;
        assert!(served.is_ok());
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
