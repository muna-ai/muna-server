/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Per-model prediction dispatcher keyed on the batch plan.
//!
//! - `Continuous`: straight to the blocking executor, fully concurrent.
//!   The compiled model owns synchronization.
//! - `Sequential`: per-model mutex so one slow model no longer blocks every other model.
//! - `Buffered`: per-model channel + one accumulator task that merges
//!   compatible requests (same batch key) up to the plan capacity, invokes
//!   once, then splits the results back per request. The accumulator holds
//!   the same per-model mutex around each invocation, because the OpenAI
//!   surfaces bypass it (muna-rs fuses translation and prediction) and take
//!   that mutex through `acquire` instead.
//!
//! Invariant: a compiled predictor that is not continuous is never invoked
//! concurrently, whichever surface the request arrives on. The guard that
//! enforces it must travel with the invocation onto the blocking thread
//! (`predict::run` / `predict::stream`), never stay in a handler future
//! that a client disconnect can drop mid-invocation.
//!
//! This module owns routing (`Dispatcher`); the buffered accumulator lives
//! in [`worker`] and the input-merge / result-split plumbing in [`merge`].

mod merge;
mod worker;

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Prediction, Value};
use muna::MunaError;

use crate::serving::batch::{compute_batch_key, item_count, BatchPlan};
use crate::serving::predict;
use crate::serving::registry::ReadyModel;
use crate::serving::stats::{PredictionSample, SampleDetail};
use worker::{BufferedWorker, PredictFn, PredictItem, CHANNEL_BUFFER};

/// Per-model dispatch state, derived from the model's `BatchPlan` on first
/// use and cached in `Dispatcher::entries` for the model's lifetime.
enum Entry {
    /// The model tolerates no concurrent invocation: requests serialize on a
    /// per-model mutex.
    Sequential { lock: Arc<tokio::sync::Mutex<()>> },
    /// Requests are queued to the model's accumulator task
    /// (`BufferedWorker`), which merges them into batches. `lock` is the
    /// model's invocation mutex: the accumulator holds it around each
    /// flush, and surfaces that bypass the accumulator take it via
    /// `acquire`.
    Buffered {
        tx: async_channel::Sender<PredictItem>,
        lock: Arc<tokio::sync::Mutex<()>>,
    },
    /// The model handles concurrency itself: requests go straight to the
    /// blocking executor with no coordination.
    Continuous,
}

/// Routes predictions through each model's batch plan.
///
/// One dispatcher serves the whole process; it is cheap to clone (a shared
/// handle). Per-model state lives in `entries`, keyed by tag and created
/// lazily on the first prediction against that model. Invocations go
/// through each model's own Muna instance (`ReadyModel::muna`), which
/// owns the loaded native handle and carries the model's deployment key.
#[derive(Clone)]
pub(crate) struct Dispatcher {
    /// Lazily-populated dispatch state per model tag.
    entries: Arc<DashMap<String, Entry>>,
}

impl Dispatcher {

    /// Create a dispatcher
    pub(crate) fn new() -> Self {
        Self { entries: Arc::new(DashMap::new()) }
    }

    /// Dispatch a raw prediction through the model's batch plan.
    pub(crate) async fn create(
        &self,
        tag: &str,
        model: &Arc<ReadyModel>,
        inputs: HashMap<String, Value>,
        acceleration: Acceleration
    ) -> Result<Prediction, MunaError> {
        self.ensure_entry(tag, model);
        // Snapshot the entry's dispatch handle without holding the shard
        // lock across an await.
        enum Route {
            Direct,
            Locked(Arc<tokio::sync::Mutex<()>>),
            Queued(async_channel::Sender<PredictItem>),
        }
        let route = match &*self.entries.get(tag).expect("entry just ensured") {
            Entry::Continuous               => Route::Direct,
            Entry::Sequential { lock }      => Route::Locked(lock.clone()),
            Entry::Buffered { tx, .. }      => Route::Queued(tx.clone()),
        };
        match route {
            Route::Direct => {
                self.predict(
                    tag,
                    model,
                    inputs,
                    acceleration,
                    None,
                    Duration::ZERO
                ).await
            }
            Route::Locked(lock) => {
                let enqueued = Instant::now();
                let guard = lock.lock_owned().await;
                self.predict(
                    tag,
                    model,
                    inputs,
                    acceleration,
                    Some(guard),
                    enqueued.elapsed()
                ).await
            }
            Route::Queued(tx) => {
                let params = match &model.plan {
                    BatchPlan::Buffered { params, .. } => params,
                    _ => unreachable!("Buffered entry implies Buffered plan"),
                };
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();
                let item = PredictItem {
                    item_count: item_count(&inputs, params),
                    batch_key: compute_batch_key(&inputs, params),
                    inputs,
                    acceleration,
                    enqueued: tokio::time::Instant::now(),
                    response_tx,
                };
                model.stats.queue_depth.fetch_add(1, Ordering::Relaxed);
                tx.send(item).await.map_err(|_| {
                    MunaError::Native("model dispatch queue closed".into())
                })?;
                response_rx.await.unwrap_or_else(|_| {
                    Err(MunaError::Native("prediction task dropped".into()))
                })
            }
        }
    }

    /// Acquire the invocation guard for a model, if its plan requires one
    /// (every plan but `Continuous`). OpenAI handlers use this around
    /// muna-rs client calls (which fuse translation and prediction,
    /// bypassing `create`); pass the result to `predict::run` / `stream`
    /// so it is released on the blocking thread, not by the handler.
    pub(crate) async fn acquire(
        &self,
        tag: &str,
        model: &Arc<ReadyModel>
    ) -> predict::Guard {
        self.ensure_entry(tag, model);
        let lock = match &*self.entries.get(tag).expect("entry just ensured") {
            Entry::Sequential { lock }      => Some(lock.clone()),
            Entry::Buffered { lock, .. }    => Some(lock.clone()),
            Entry::Continuous               => None,
        }?;
        Some(lock.lock_owned().await)
    }

    /// Drop a model's dispatch entry (closes the accumulator task, if any).
    pub(crate) fn remove(&self, tag: &str) {
        self.entries.remove(tag);
    }

    fn ensure_entry(&self, tag: &str, model: &Arc<ReadyModel>) {
        if self.entries.contains_key(tag) {
            return;
        }
        let entry = match &model.plan {
            BatchPlan::Sequential => Entry::Sequential {
                lock: Arc::new(tokio::sync::Mutex::new(()))
            },
            BatchPlan::Continuous => Entry::Continuous,
            BatchPlan::Buffered { params, capacity } => {
                let (tx, rx) = async_channel::bounded(CHANNEL_BUFFER);
                let lock = Arc::new(tokio::sync::Mutex::new(()));
                let muna = model.muna.clone();
                let tag_owned = tag.to_string();
                let stats = model.stats.clone();
                let invoke_lock = lock.clone();
                let predict_fn: PredictFn = Arc::new(move |inputs, acceleration| {
                    let muna = muna.clone();
                    let tag = tag_owned.clone();
                    let stats = stats.clone();
                    let lock = invoke_lock.clone();
                    Box::pin(async move {
                        // The accumulator is already single-consumer; the
                        // lock only contends with `acquire` callers on the
                        // OpenAI surfaces.
                        let guard = lock.lock_owned().await;
                        predict::run(Some(guard), move || async move {
                            let start = Instant::now();
                            let result = muna.predictions.create(
                                &tag,
                                Some(inputs),
                                Some(acceleration),
                                None,
                                None
                            ).await;
                            stats.record_latency(start.elapsed());
                            result
                        }).await
                    })
                });
                let worker = BufferedWorker {
                    stats: model.stats.clone(),
                    params: params.clone(),
                    capacity: *capacity,
                    predict_fn,
                    rx,
                };
                tokio::spawn(worker.run());
                Entry::Buffered { tx, lock }
            }
        };
        self.entries.entry(tag.to_string()).or_insert(entry);
    }

    async fn predict(
        &self,
        tag: &str,
        model: &Arc<ReadyModel>,
        inputs: HashMap<String, Value>,
        acceleration: Acceleration,
        guard: predict::Guard,
        queue_wait: Duration
    ) -> Result<Prediction, MunaError> {
        let muna = model.muna.clone();
        let tag_owned = tag.to_string();
        let stats = model.stats.clone();
        predict::run(guard, move || async move {
            let start = Instant::now();
            let result = muna.predictions.create(
                &tag_owned,
                Some(inputs),
                Some(acceleration),
                None,
                None
            ).await;
            let latency = start.elapsed();
            stats.record_latency(latency);
            stats.telemetry.record(PredictionSample {
                at: Instant::now(),
                queue_wait,
                latency,
                detail: SampleDetail::Unary,
            });
            result
        }).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use muna::types::Signature;
    use muna::Muna;

    use super::*;
    use crate::serving::stats::ModelStats;

    fn ready_model(plan: BatchPlan) -> Arc<ReadyModel> {
        Arc::new(ReadyModel {
            muna: Arc::new(Muna::new(None, None)),
            loaded_at: Instant::now(),
            signature: Signature { inputs: vec![], outputs: vec![] },
            plan,
            stats: Arc::new(ModelStats::new()),
        })
    }

    fn buffered_plan() -> BatchPlan {
        BatchPlan::Buffered {
            params: HashSet::from(["prompt".to_string()]),
            capacity: 4,
        }
    }

    /// Buffered models (FLUX.2 [klein], the embedders) reach the predictor
    /// through the OpenAI surfaces via `acquire`, bypassing the accumulator;
    /// they must get the model's invocation guard, not `None`.
    #[tokio::test]
    async fn acquire_guards_buffered_models() {
        let dispatcher = Dispatcher::new();
        let model = ready_model(buffered_plan());
        let first = dispatcher.acquire("@test/buffered", &model).await;
        assert!(first.is_some(), "buffered plan must yield an invocation guard");
        // A second caller blocks until the first guard is released.
        let second = tokio::time::timeout(
            Duration::from_millis(50),
            dispatcher.acquire("@test/buffered", &model)
        ).await;
        assert!(second.is_err(), "second acquire must wait on the first");
        drop(first);
        let second = tokio::time::timeout(
            Duration::from_secs(1),
            dispatcher.acquire("@test/buffered", &model)
        ).await.expect("acquire after release");
        assert!(second.is_some());
    }

    #[tokio::test]
    async fn acquire_guards_sequential_models() {
        let dispatcher = Dispatcher::new();
        let model = ready_model(BatchPlan::Sequential);
        assert!(dispatcher.acquire("@test/sequential", &model).await.is_some());
    }

    /// Continuous predictors own their synchronization; no guard.
    #[tokio::test]
    async fn acquire_skips_continuous_models() {
        let dispatcher = Dispatcher::new();
        let model = ready_model(BatchPlan::Continuous);
        assert!(dispatcher.acquire("@test/continuous", &model).await.is_none());
    }

    /// The accumulator and `acquire` callers share ONE lock per model, so a
    /// raw `/v1/predictions` batch can never overlap an OpenAI-surface call.
    #[tokio::test]
    async fn buffered_accumulator_shares_the_acquire_lock() {
        let dispatcher = Dispatcher::new();
        let model = ready_model(buffered_plan());
        let tag = "@test/buffered";
        dispatcher.ensure_entry(tag, &model);
        let lock = match &*dispatcher.entries.get(tag).expect("entry") {
            Entry::Buffered { lock, .. } => lock.clone(),
            _ => panic!("expected a buffered entry"),
        };
        // `acquire` hands out the very same mutex the accumulator holds.
        let guard = dispatcher.acquire(tag, &model).await.expect("guard");
        assert!(lock.try_lock().is_err());
        drop(guard);
        assert!(lock.try_lock().is_ok());
    }
}
