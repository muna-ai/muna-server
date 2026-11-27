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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Device, Prediction, Signature, Value};
use muna::Muna;

use crate::serving::prepared::PreparedModels;
use crate::serving::preparation::{FailureStage, PreparationTracker};
use crate::platform::Platform;
use crate::notifications::NotificationCenter;
use crate::serving::batch::BatchPlan;
use crate::serving::predict;
use crate::serving::stats::ModelStats;
use crate::state::{DeviceStore, KeyStore, PolicyStore};
use crate::serving::lease::{BusyGuard, LeaseSupervisor, ModelLease};

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
    /// Deployment whose metadata and native engine this handle owns.
    pub key_hash: Option<String>,
    /// Native execution must drain before this assignment can be retired.
    pub execution: Arc<ModelExecution>,
    pub lease: Option<ModelLease>,
    /// When the load completed (reported as the model's `created` age by
    /// the OpenAI-compatible `/v1/models` handler).
    pub loaded_at: Instant,
    /// The predictor signature: the batch plan derives from it, and the
    /// `/v1/models` rows advertise its limits and capabilities.
    pub signature: Signature,
    pub adapters: super::adapters::Adapters,
    /// How the dispatcher batches predictions for this model, derived from
    /// the signature's batch config.
    pub plan: BatchPlan,
    /// Live counters (queue depth, prediction totals, latency, load time,
    /// VRAM), shared with the dispatcher and reported in heartbeats.
    pub stats: Arc<ModelStats>,
    /// Explicit GPU ordinals the engine was loaded on, in rank order.
    /// Production GPU loads reject missing or empty assignments.
    pub devices: Option<Vec<u32>>,
    /// Content hash of the loaded predictor build; see [`build_id`].
    pub build: Option<String>,
}

/// Busy intent drops before the execution read lock permits engine deletion.
pub(crate) struct ModelAdmission {
    _busy: Option<BusyGuard>,
    _execution: tokio::sync::OwnedRwLockReadGuard<()>,
}

impl ReadyModel {
    pub(crate) async fn admit(&self) -> Result<ModelAdmission, muna::MunaError> {
        let execution = self.execution.enter().await?;
        let busy = self.lease.as_ref().map(|lease| lease.enter()).transpose()?;
        Ok(ModelAdmission { _busy: busy, _execution: execution })
    }

    pub(crate) fn acceleration(&self) -> Acceleration {
        if self.devices.as_ref().is_some_and(Vec::is_empty) { Acceleration::LocalCpu }
        else { Acceleration::LocalGpu }
    }

    pub(crate) fn device_handles(&self) -> Result<Option<Vec<Device>>, muna::MunaError> {
        Ok(self.devices.as_ref().map(|ids| ids.iter().map(|id| Device::cuda(*id as i32)).collect()))
    }
}

/// One loaded engine's execution lifetime, shared by unary and streaming calls.
#[derive(Default)]
pub(crate) struct ModelExecution {
    gate: Arc<tokio::sync::RwLock<()>>,
    retired: AtomicBool,
}

impl ModelExecution {

    pub(crate) async fn enter(&self) -> Result<tokio::sync::OwnedRwLockReadGuard<()>, muna::MunaError> {
        // The only writer retires this assignment. Never queue a second
        // read behind it while an admitted request already owns a read:
        // that would prevent both the invocation and retirement progressing.
        let guard = self.gate.clone().try_read_owned().map_err(|_|
            muna::MunaError::Native("Model device assignment retiring; retry the request.".into()))?;
        if self.retired.load(Ordering::Acquire) {
            return Err(muna::MunaError::Native("Model device assignment retired; retry the request.".into()));
        }
        Ok(guard)
    }

    pub(crate) fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }

    pub(crate) fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }
}

/// Lifecycle state of one model. Absent-from-map is the implicit fourth
/// state (never requested, or unloaded); `Failed` is cleared on read so the
/// next request retries the load.
pub(crate) enum ModelState {
    /// Single-flight warmup in progress since the given instant (used to
    /// derive `Retry-After` for requests that give up waiting), on the
    /// given devices.
    Loading { since: Instant, devices: Option<Vec<u32>>, key_hash: Option<String> },
    /// Engine warm, signature and batch plan known; shared with every
    /// request currently predicting against the model.
    Ready(Arc<ReadyModel>),
    /// The warmup failed at the given instant; `error` is surfaced to the
    /// requester (and the control plane) verbatim.
    Failed { error: String, at: Instant, key_hash: Option<String> },
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
    /// Content hash of the predictor build; see [`build_id`].
    pub build: Option<String>,
}

/// Loader delegate: performs the warmup + signature fetch for one tag on
/// the given devices. Injectable so single-flight behavior is testable
/// without a live engine.
///
/// The delegate promotes queued resources before spawning preparation; the
/// resource scheduler owns transfers independently of model loading.
type Loader = Arc<
    dyn Fn(String, Option<Vec<u32>>)
        -> futures_util::future::BoxFuture<'static, Result<Loaded, String>>
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
    /// Control-plane device assignments, read when a load starts.
    devices: DeviceStore,
    keys: KeyStore,
    /// Pinned model set (`--models`): the only tags this server will load.
    /// `None` means open behavior (any tag loadable on demand).
    pinned: Option<HashSet<String>>,
    /// Most recent successful load duration in seconds, for Retry-After.
    last_load_secs: AtomicU64,
    /// Poked on every lifecycle write so the heartbeat reports it at once.
    notifications: Arc<NotificationCenter>,
    load_events: Option<Arc<crate::control::events::LoadEvents>>,
    /// Device memory source for the load-time VRAM delta.
    platform: Arc<dyn Platform>,
    lease: Option<Arc<LeaseSupervisor>>,
    policies: PolicyStore,
}

impl ModelRegistry {

    /// Create a model registry. `keys` holds the per-tag deployment keys
    /// delivered by control-plane residency directives; tags without one
    /// load through the process-wide `$MUNA_ACCESS_KEY`. `devices` holds
    /// the per-tag device assignments; tags without one load on the
    /// engine's default devices.
    pub(crate) fn new(
        keys: KeyStore,
        devices: DeviceStore,
        pinned: Option<HashSet<String>>,
        notifications: Arc<NotificationCenter>,
        platform: Arc<dyn Platform>,
        lease: Option<Arc<LeaseSupervisor>>,
        policies: PolicyStore,
        preparation: PreparationTracker,
        prepared: PreparedModels
    ) -> Self {
        let identities = keys.clone();
        let load_lease = lease.clone();
        let load_policies = policies.clone();
        let loader: Loader = Arc::new(move |tag, devices| {
            prepared.set_process(&tag, true);
            let key = keys.get(&tag).map(|entry| entry.value().clone());
            let lease = load_lease.clone();
            let preparation = preparation.clone();
            let prepared = prepared.clone();
            let dedicated = load_policies.get(&tag).is_some_and(|p| p.dedicated);
            Box::pin(async move {
                let loaded = load_model(&tag, key, devices.clone(), preparation, prepared).await?;
                if let Some(lease) = &lease {
                    // CPU predictors need no device registration. Managed GPU
                    // assignments always require an exact Corby group match.
                    if let Some(assigned) = devices.clone().filter(|ids| !ids.is_empty()).or_else(|| lease.devices_for(&tag)) {
                        if let Err(error) = lease.configure(&tag, &assigned, dedicated) {
                            delete_predictor(&loaded.muna, &tag).await;
                            return Err(error);
                        }
                    } else if dedicated {
                        delete_predictor(&loaded.muna, &tag).await;
                        return Err("Dedicated predictors must register with Corby.".into());
                    }
                }
                Ok(loaded)
            })
        });
        let mut registry = Self::with_loader(loader, devices, pinned, notifications, platform);
        let inner = Arc::get_mut(&mut registry.inner).unwrap();
        inner.lease = lease;
        inner.keys = identities;
        inner.policies = policies;
        registry
    }

    fn with_loader(
        loader: Loader,
        devices: DeviceStore,
        pinned: Option<HashSet<String>>,
        notifications: Arc<NotificationCenter>,
        platform: Arc<dyn Platform>
    ) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                models: DashMap::new(),
                loader,
                devices,
                keys: Arc::default(),
                pinned,
                last_load_secs: AtomicU64::new(DEFAULT_LOAD_SECS),
                notifications,
                load_events: None,
                platform,
                lease: None,
                policies: Arc::new(DashMap::new())
            })
        }
    }

    pub(crate) fn with_load_events(mut self, events: Option<Arc<crate::control::events::LoadEvents>>) -> Self {
        Arc::get_mut(&mut self.inner).expect("registry already shared").load_events = events;
        self
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
                    ModelState::Ready(model) if !model.execution.is_retired() => return Ok(model.clone()),
                    ModelState::Ready(_) => return Err(RegistryError::Loading { retry_after: 1 }),
                    ModelState::Failed { error, .. } => {
                        let error = error.clone();
                        entry.remove();
                        return Err(RegistryError::Failed(error));
                    }
                    ModelState::Loading { since, .. } => {
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
                    let devices = self.assigned_devices(tag);
                    entry.insert(Slot {
                        state: ModelState::Loading { since: Instant::now(), devices: devices.clone(), key_hash: self.current_key_hash(tag) },
                        watch,
                        unload_requested: false
                    });
                    self.inner.notifications.status_changed();
                    self.spawn_load(tag.to_string(), devices);
                    rx
                }
            };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if tokio::time::timeout(remaining, waiter.changed()).await.is_err() {
                let since = self.inner.models.get(tag).and_then(|slot| match &slot.state {
                    ModelState::Loading { since, .. } => Some(*since),
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
                let devices = self.assigned_devices(tag);
                entry.insert(Slot {
                    state: ModelState::Loading { since: Instant::now(), devices: devices.clone(), key_hash: self.current_key_hash(tag) },
                    watch,
                    unload_requested: false
                });
                self.inner.notifications.status_changed();
                self.spawn_load(tag.to_string(), devices);
            }
        }
    }

    /// Devices the model's engine is loading or loaded on: `None` when the
    /// tag has no engine (absent or failed), `Some(None)` when the engine
    /// picked its own devices.
    pub(crate) fn devices(&self, tag: &str) -> Option<Option<Vec<u32>>> {
        self.inner.models.get(tag).and_then(|slot| match &slot.state {
            ModelState::Loading { devices, .. } => Some(devices.clone()),
            ModelState::Ready(model) => Some(model.devices.clone()),
            ModelState::Failed { .. } => None,
        })
    }

    fn current_key_hash(&self, tag: &str) -> Option<String> {
        self.inner.keys.get(tag).map(|key| crate::client::key_hash(key.value()))
    }

    fn assigned_devices(&self, tag: &str) -> Option<Vec<u32>> {
        self.inner.devices.get(tag).map(|entry| entry.value().clone())
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

    /// Stop new admissions synchronously, before waiting on any engine drain.
    pub(crate) fn retire(&self, tag: &str) {
        if let Some(mut slot) = self.inner.models.get_mut(tag) {
            slot.unload_requested = true;
            if let ModelState::Ready(model) = &slot.state {
                model.execution.retire();
            }
        }
        self.inner.notifications.status_changed();
    }

    /// Drain native execution and delete through the owning SDK client. An
    /// initializing engine is deleted by its load task; wait for that deletion
    /// before making the tag available for another deployment.
    pub(crate) async fn unload(&self, tag: &str) {
        let (model, mut completion) = {
            let Some(mut slot) = self.inner.models.get_mut(tag) else { return; };
            match &slot.state {
                ModelState::Loading { .. } => {
                    slot.unload_requested = true;
                    (None, Some(slot.watch.subscribe()))
                }
                ModelState::Ready(model) => (Some(model.clone()), None),
                ModelState::Failed { .. } => (None, None),
            }
        };
        if let Some(waiter) = &mut completion {
            // Native initialization is not cancellable. Its task deletes the
            // resulting handle before closing the slot, then replacement is safe.
            while waiter.changed().await.is_ok() {}
            return;
        }
        let Some(model) = model else {
            self.inner.models.remove(tag);
            self.inner.notifications.status_changed();
            return;
        };
        model.execution.retire();
        self.inner.notifications.status_changed();
        // Keep the slot occupied until the old engine is gone. Neither a request
        // nor the next heartbeat can start a second copy during this drain.
        let _exclusive = model.execution.gate.clone().write_owned().await;
        if !self.inner.models.get(tag).is_some_and(|slot|
            matches!(&slot.state, ModelState::Ready(current) if Arc::ptr_eq(current, &model))) {
            return;
        }
        delete_predictor(&model.muna, tag).await;
        self.inner.models.remove(tag);
        self.inner.notifications.status_changed();
    }

    /// Snapshot every slot's state for status reporting.
    pub(crate) fn snapshot(&self) -> Vec<(String, ModelState)> {
        self.inner.models.iter()
            .map(|entry| {
                let state = match &entry.state {
                    ModelState::Loading { since, devices, key_hash } => ModelState::Loading {
                        since: *since,
                        devices: devices.clone(),
                        key_hash: key_hash.clone()
                    },
                    ModelState::Ready(model) => ModelState::Ready(model.clone()),
                    ModelState::Failed { error, at, key_hash } => ModelState::Failed {
                        error: error.clone(),
                        at: *at,
                        key_hash: key_hash.clone()
                    },
                };
                (entry.key().clone(), state)
            })
            .collect()
    }

    /// Tags currently in the `Ready` state.
    pub(crate) fn ready_tags(&self) -> Vec<String> {
        self.inner.models.iter()
            .filter(|entry| matches!(&entry.state, ModelState::Ready(model) if !model.execution.is_retired()))
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// The model's `Ready` handle, if currently loaded. Never triggers a
    /// load (unlike `ensure_ready`).
    pub(crate) fn ready(&self, tag: &str) -> Option<Arc<ReadyModel>> {
        self.inner.models.get(tag).and_then(|slot| match &slot.state {
            ModelState::Ready(model) if !model.execution.is_retired() => Some(model.clone()),
            _ => None,
        })
    }

    fn retry_after(&self, since: Instant) -> u64 {
        let expected = self.inner.last_load_secs.load(Ordering::Relaxed);
        expected.saturating_sub(since.elapsed().as_secs()).max(5)
    }

    /// Spawn the single-flight warmup: sentinel prediction (reaches the
    /// engine's `load_predictor`), then plan derivation from its prepared signature.
    ///
    /// Invoking the loader synchronously promotes this tag's queued resources
    /// before subsequently scheduled disk-only preparations can start.
    fn spawn_load(
        &self,
        tag: String,
        devices: Option<Vec<u32>>
    ) {
        let key_hash = self.current_key_hash(&tag);
        let inner = self.inner.clone();
        let loading = (inner.loader)(tag.clone(), devices.clone());
        tokio::spawn(async move {
            let start = Instant::now();
            // Measured over the assigned devices only, so a concurrent load
            // on other GPUs does not pollute this model's delta.
            let memory_used_mb = || match &devices {
                Some(devices) => inner.platform.memory_used_mb_on(devices),
                None => inner.platform.memory_used_mb(),
            };
            let vram_before = memory_used_mb();
            let stats = Arc::new(ModelStats::new());
            let outcome = loading.await;
            let completed = std::time::SystemTime::now().into();
            let vram_after = memory_used_mb();
            if let (Some(before), Some(after)) = (vram_before, vram_after) {
                if after > before {
                    stats.record_vram(after - before);
                }
            }
            // Kept out of `state` so a deferred unload can delete through
            // the instance that actually loaded the handle.
            let mut loaded_muna: Option<Arc<Muna>> = None;
            let state = match outcome {
                Ok(Loaded { signature, muna, download, load, build }) => {
                    if let Some(events) = &inner.load_events {
                        let gpus = inner.platform.devices();
                        events.record(crate::control::events::ModelLoadEvent {
                            id: uuid::Uuid::new_v4().to_string(),
                            tag: tag.clone(),
                            created: completed,
                            duration_ms: load.as_secs_f64() * 1000.0,
                            download_ms: download.as_secs_f64() * 1000.0,
                            server_version: env!("CARGO_PKG_VERSION").to_string(),
                            device_ids: devices.clone(),
                            build: build.clone(),
                            vram_mb: stats.vram_mb(),
                            gpu_family: gpus.iter().find_map(|gpu| gpu.family)
                                .and_then(|family| serde_json::to_value(family).ok())
                                .and_then(|value| value.as_str().map(str::to_owned)),
                            gpu_name: gpus.first().map(|gpu| gpu.name.clone()),
                            concurrent_loads: inner.models.iter().filter(|entry| {
                                entry.key() != &tag && matches!(entry.value().state, ModelState::Loading { .. })
                            }).count() as u32,
                        });
                    }
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
                        lease: inner.lease.as_ref().and_then(|supervisor| {
                            let assigned = supervisor.devices_for(&tag)?;
                            Some(ModelLease {
                                supervisor: supervisor.clone(),
                                label: tag.clone(),
                                devices: devices.clone().unwrap_or(assigned),
                                dedicated: inner.policies.get(&tag).is_some_and(|p| p.dedicated),
                            })
                        }),
                        execution: Arc::default(),
                        key_hash: key_hash.clone(),
                        muna,
                        loaded_at: Instant::now(),
                        plan: BatchPlan::from_signature(&signature),
                        adapters: super::adapters::Adapters::new(signature.clone()),
                        signature,
                        stats,
                        devices,
                        build
                    }))
                }
                Err(error) => {
                    tracing::error!(tag = %tag, error = %error, "model load failed");
                    ModelState::Failed { error, at: Instant::now(), key_hash }
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
                if let Some(muna) = loaded_muna {
                    delete_predictor(&muna, &tag).await;
                }
                inner.models.remove(&tag);
            }
            // Ready, Failed, or gone: all three are reportable transitions.
            inner.notifications.status_changed();
        });
    }
}

async fn delete_predictor(muna: &Arc<Muna>, tag: &str) {
    let delete_muna = muna.clone();
    let delete_tag = tag.to_string();
    let result = predict::run(None, move || async move {
        delete_muna.predictions.delete(&delete_tag).await
    }).await;
    match result {
        Ok(_) => tracing::info!(tag = %tag, "model unloaded"),
        Err(e) => tracing::warn!(tag = %tag, error = %e, "failed to unload predictor"),
    }
}

async fn load_model(
    tag: &str,
    key: Option<String>,
    devices: Option<Vec<u32>>,
    preparation: PreparationTracker,
    prepared: PreparedModels
) -> Result<Loaded, String> {
    if let Some(ids) = &devices {
        let unique: HashSet<_> = ids.iter().copied().collect();
        if unique.len() != ids.len() || ids.iter().any(|id| *id > i32::MAX as u32) {
            return Err("GPU device assignments must contain distinct valid ordinals.".into());
        }
    }
    let acceleration = if devices.as_ref().is_some_and(Vec::is_empty) {
        Acceleration::LocalCpu
    } else if devices.is_some() { Acceleration::LocalGpu } else { Acceleration::LocalAuto };
    // A completed disk preparation already localized the resources and fetched
    // the signature. Reuse that keyed client; native loading below still owns
    // token validation and refresh through the SDK's persistent cache.
    let download_started = Instant::now();
    let prepared = prepared.prepare(tag, key, preparation.clone()).await?;
    let preparation = preparation.scoped(tag);
    // A fresh service owns exactly this engine's handles; the shared prepared
    // snapshot retains only its HTTP transport and cannot extend engine life.
    let muna = Arc::new(Muna::with_client(prepared.client.clone()));
    let downloaded = &prepared.prediction;
    let required = downloaded.requirements.as_ref().and_then(|r| r.gpu_count).unwrap_or(0);
    if devices.as_ref().is_some_and(|ids| ids.len() < required as usize) {
        return Err(format!("Predictor requires {required} GPUs; the assigned device list is insufficient."));
    }
    preparation.cached(tag);
    let download = download_started.elapsed();
    let build = build_id(downloaded);
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
    // The control plane assigns these from the heartbeat's NVML indices
    // and they are used here as CUDA ordinals. The two agree only while
    // CUDA_VISIBLE_DEVICES is unset and every GPU on the node is the same
    // model (CUDA's fastest-first order then ties back to PCI bus order).
    let handles = devices.map(|ids| ids.into_iter().map(|index| Device::cuda(index as i32)).collect::<Vec<_>>());
    predict::run(None, move || async move {
        let inputs = HashMap::from([("_".to_string(), Value::Null)]);
        warm_muna.predictions.create(
            &warm_tag,
            Some(inputs),
            Some(acceleration),
            handles.as_deref(),
            None,
            None,
            None,
            None
        ).await
    }).await.map_err(|e| { preparation.failed(tag, FailureStage::Load); e.to_string() })?;
    preparation.cached(tag);
    let load = load_started.elapsed();
    Ok(Loaded { signature: prepared.signature.clone(), muna, download, load, build })
}

/// Content hash of a prediction's build: the URL path stem of its last
/// `dso` resource, which is always the compiled predictor (the rest are
/// shared runtime libraries). Resource URLs are checksum-named, so the
/// stem changes exactly when the predictor binary does.
fn build_id(prediction: &Prediction) -> Option<String> {
    let resource = prediction
        .resources
        .as_ref()?
        .iter()
        .rev()
        .find(|resource| resource.kind == "dso")?;
    let path = resource.url.split(['?', '#']).next()?;
    let name = path.rsplit(['/', '\\']).next()?;
    let stem = std::path::Path::new(name).file_stem()?.to_str()?;
    (!stem.is_empty()).then(|| stem.to_string())
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
        registry_notifying(
            loads,
            fail,
            delay,
            pinned,
            Arc::new(NotificationCenter::default())
        )
    }

    fn registry_notifying(
        loads: Arc<AtomicUsize>,
        fail: bool,
        delay: Duration,
        pinned: Option<HashSet<String>>,
        notifications: Arc<NotificationCenter>
    ) -> ModelRegistry {
        let loader: Loader = Arc::new(move |_tag, _devices| {
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
                        load: delay,
                        build: None
                    })
                }
            })
        });
        ModelRegistry::with_loader(
            loader,
            Arc::new(DashMap::new()),
            pinned,
            notifications,
            Arc::new(crate::platform::NullPlatform)
        )
    }

    #[tokio::test]
    async fn fast_loads_and_same_build_reloads_each_emit_a_completion() {
        let path = std::env::temp_dir().join(format!("muna-registry-events-{}", uuid::Uuid::new_v4()));
        let notifications = Arc::new(NotificationCenter::default());
        let events = Arc::new(crate::control::events::LoadEvents::open(&path, "node", notifications.clone()).unwrap());
        let shutdown = tokio_util::sync::CancellationToken::new();
        let writer = tokio::spawn({
            let events = events.clone();
            let shutdown = shutdown.clone();
            async move { events.run(&shutdown).await }
        });
        let registry = registry_notifying(Arc::default(), false, Duration::from_micros(125), None, notifications.clone())
            .with_load_events(Some(events.clone()));
        registry.ensure_ready("@test/fast").await.unwrap();
        registry.unload("@test/fast").await;
        // The first fact survives after its model has disappeared entirely.
        assert!(registry.snapshot().is_empty());
        registry.ensure_ready("@test/fast").await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while events.batch().len() != 2 { notifications.status.notified().await; }
        }).await.unwrap();
        let batch = events.batch();
        assert_ne!(batch[0].id, batch[1].id);
        assert_eq!(batch[0].duration_ms, 0.125);
        assert_eq!(batch[1].duration_ms, 0.125);
        registry.unload("@test/fast").await;
        shutdown.cancel();
        writer.await.unwrap();
        assert_eq!(crate::control::events::LoadEvents::open(&path, "node", Arc::default()).unwrap().batch().len(), 2);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn failed_load_does_not_emit_a_successful_completion() {
        let path = std::env::temp_dir().join(format!("muna-registry-events-{}", uuid::Uuid::new_v4()));
        let events = Arc::new(crate::control::events::LoadEvents::open(&path, "node", Arc::default()).unwrap());
        let registry = registry_with(Arc::default(), true, Duration::ZERO).with_load_events(Some(events.clone()));
        assert!(registry.ensure_ready("@test/failed").await.is_err());
        let shutdown = tokio_util::sync::CancellationToken::new();
        shutdown.cancel();
        events.clone().run(&shutdown).await;
        assert!(events.batch().is_empty());
        std::fs::remove_dir_all(path).unwrap();
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
        tokio::time::timeout(Duration::from_secs(5), async {
            while !registry.snapshot().is_empty() {
                assert!(registry.ready_tags().is_empty());
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }).await.expect("deferred deletion must finish before the slot is freed");
        assert!(registry.ready_tags().is_empty());
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

    /// Every lifecycle write must wake the heartbeat: one permit after the
    /// `Loading` insert, another once the load resolves to `Ready`.
    #[tokio::test]
    async fn lifecycle_transitions_notify_status() {
        let loads = Arc::new(AtomicUsize::new(0));
        let notifications = Arc::new(NotificationCenter::default());
        let registry = registry_notifying(
            loads,
            false,
            Duration::from_millis(50),
            None,
            notifications.clone()
        );
        let observed = |what: &'static str| {
            let notifications = notifications.clone();
            async move {
                tokio::time::timeout(Duration::from_millis(500), notifications.status.notified())
                    .await
                    .unwrap_or_else(|_| panic!("no status notification after {what}"));
            }
        };
        registry.warm("@test/model");
        observed("Loading").await;
        // The load is still in flight (50 ms); the next permit is Ready.
        observed("Ready").await;
        assert_eq!(registry.ready_tags(), vec!["@test/model".to_string()]);
        registry.unload("@test/model").await;
        observed("unload").await;
    }

    /// The loader receives the device assignment present when the load
    /// starts, and the registry reports it through `Loading` and `Ready`.
    #[tokio::test]
    async fn loads_on_assigned_devices() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let loader_seen = seen.clone();
        let loader: Loader = Arc::new(move |_tag, devices| {
            loader_seen.lock().unwrap().push(devices);
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(Loaded {
                    signature: Signature { inputs: vec![], outputs: vec![] },
                    muna: Arc::new(Muna::new(None, None)),
                    download: Duration::ZERO,
                    load: Duration::ZERO,
                    build: Some("abc123".to_string())
                })
            })
        });
        let devices: DeviceStore = Arc::new(DashMap::new());
        devices.insert("@test/model".to_string(), vec![4, 5]);
        devices.insert("@test/cpu".to_string(), vec![]);
        let registry = ModelRegistry::with_loader(
            loader,
            devices,
            None,
            Arc::new(NotificationCenter::default()),
            Arc::new(crate::platform::NullPlatform)
        );
        assert_eq!(registry.devices("@test/model"), None);
        registry.warm("@test/model");
        assert_eq!(registry.devices("@test/model"), Some(Some(vec![4, 5])));
        let model = registry.ensure_ready("@test/model").await.unwrap();
        assert_eq!(model.devices, Some(vec![4, 5]));
        assert_eq!(model.build.as_deref(), Some("abc123"));
        assert_eq!(model.acceleration(), Acceleration::LocalGpu);
        assert_eq!(*seen.lock().unwrap(), vec![Some(vec![4, 5])]);
        let cpu = registry.ensure_ready("@test/cpu").await.unwrap();
        assert_eq!(cpu.devices, Some(vec![]));
        assert_eq!(cpu.acceleration(), Acceleration::LocalCpu);
        assert!(cpu.device_handles().unwrap().unwrap().is_empty());
        assert!(cpu.lease.is_none());
        // Unassigned tags load on the engine default.
        registry.warm("@test/other");
        assert_eq!(registry.devices("@test/other"), Some(None));
    }

    #[test]
    fn build_id_is_last_dso_url_stem() {
        let resource = |kind: &str, url: &str| muna::types::PredictionResource {
            kind: kind.to_string(),
            url: url.to_string(),
            name: None,
            size: None
        };
        let prediction = |resources| Prediction {
            id: "pred".to_string(),
            tag: "@test/model".to_string(),
            created: String::new(),
            configuration: None,
            resources: Some(resources),
            requirements: None,
            results: None,
            latency: None,
            error: None,
            logs: None
        };
        let signed = prediction(vec![
            resource("dso", "https://cdn.muna.ai/resources/cudart?sig=x"),
            resource("bin", "https://cdn.muna.ai/resources/weights"),
            resource("dso", "https://cdn.muna.ai/resources/9f2c41?X-Amz-Signature=y")
        ]);
        assert_eq!(build_id(&signed).as_deref(), Some("9f2c41"));
        let local = prediction(vec![resource("dso", "/cache/hf/snapshots/abc/9f2c41.so")]);
        assert_eq!(build_id(&local).as_deref(), Some("9f2c41"));
        assert_eq!(build_id(&prediction(vec![resource("bin", "https://cdn/x")])), None);
    }
    #[tokio::test]
    async fn device_move_drains_execution_before_allowing_a_replacement() {
        let loads = Arc::new(AtomicUsize::new(0));
        let registry = registry_with(loads.clone(), false, Duration::ZERO);
        let model = registry.ensure_ready("@test/move").await.unwrap();
        let execution = model.execution.enter().await.unwrap();
        let retiring = registry.clone();
        let unload = tokio::spawn(async move { retiring.unload("@test/move").await });
        while !model.execution.is_retired() { tokio::task::yield_now().await; }
        assert!(!unload.is_finished());
        assert!(matches!(registry.ensure_ready("@test/move").await, Err(RegistryError::Loading { .. })));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        let retry = tokio::time::timeout(Duration::from_millis(100), model.execution.enter()).await;
        assert!(matches!(retry, Ok(Err(_))), "new native admission must not wait behind retirement while a caller owns an admission guard");
        drop(execution);
        unload.await.unwrap();
        assert!(model.execution.enter().await.is_err());
        registry.ensure_ready("@test/move").await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

}

#[cfg(test)]
#[path = "registry_rollout_tests.rs"]
mod rollout_tests;
