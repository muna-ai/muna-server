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
//!   once, then splits the results back per request.
//!
//! This module owns routing (`Dispatcher`); the buffered accumulator lives
//! in [`worker`] and the input-merge / result-split plumbing in [`merge`].

mod merge;
mod worker;

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use muna::types::{Acceleration, Prediction, Value};
use muna::{Muna, MunaError};

use crate::serving::batch::{compute_batch_key, item_count, BatchPlan};
use crate::serving::predict;
use crate::serving::registry::ReadyModel;
use worker::{BufferedWorker, PredictFn, PredictItem, CHANNEL_BUFFER};

/// Per-model dispatch state, derived from the model's `BatchPlan` on first
/// use and cached in `Dispatcher::entries` for the model's lifetime.
enum Entry {
    /// The model tolerates no concurrent invocation: requests serialize on a
    /// per-model mutex.
    Sequential { lock: Arc<tokio::sync::Mutex<()>> },
    /// Requests are queued to the model's accumulator task
    /// (`BufferedWorker`), which merges them into batches.
    Buffered { tx: async_channel::Sender<PredictItem> },
    /// The model handles concurrency itself: requests go straight to the
    /// blocking executor with no coordination.
    Continuous,
}

/// Routes predictions through each model's batch plan.
///
/// One dispatcher serves the whole process; it is cheap to clone (both
/// fields are shared handles). Per-model state lives in `entries`, keyed by
/// tag and created lazily on the first prediction against that model.
#[derive(Clone)]
pub(crate) struct Dispatcher {
    /// Client through which invocations are made.
    muna: Arc<Muna>,
    /// Lazily-populated dispatch state per model tag.
    entries: Arc<DashMap<String, Entry>>,
}

impl Dispatcher {

    /// Create a dispatcher
    pub(crate) fn new(muna: Arc<Muna>) -> Self {
        Self { muna, entries: Arc::new(DashMap::new()) }
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
            Entry::Continuous => Route::Direct,
            Entry::Sequential { lock } => Route::Locked(lock.clone()),
            Entry::Buffered { tx } => Route::Queued(tx.clone()),
        };
        match route {
            Route::Direct => self.predict(tag, model, inputs, acceleration).await,
            Route::Locked(lock) => {
                let _guard = lock.lock().await;
                self.predict(tag, model, inputs, acceleration).await
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

    /// Acquire the sequential guard for a model, if its plan requires one.
    /// OpenAI handlers use this around muna-rs client calls (which fuse
    /// translation and prediction, bypassing `create`).
    pub(crate) async fn acquire(
        &self,
        tag: &str,
        model: &Arc<ReadyModel>
    ) -> Option<tokio::sync::OwnedMutexGuard<()>> {
        self.ensure_entry(tag, model);
        let lock = match &*self.entries.get(tag).expect("entry just ensured") {
            Entry::Sequential { lock } => Some(lock.clone()),
            _ => None,
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
                let muna = self.muna.clone();
                let tag_owned = tag.to_string();
                let stats = model.stats.clone();
                let predict_fn: PredictFn = Arc::new(move |inputs, acceleration| {
                    let muna = muna.clone();
                    let tag = tag_owned.clone();
                    let stats = stats.clone();
                    Box::pin(async move {
                        predict::run(move || async move {
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
                Entry::Buffered { tx }
            }
        };
        self.entries.entry(tag.to_string()).or_insert(entry);
    }

    async fn predict(
        &self,
        tag: &str,
        model: &Arc<ReadyModel>,
        inputs: HashMap<String, Value>,
        acceleration: Acceleration
    ) -> Result<Prediction, MunaError> {
        let muna = self.muna.clone();
        let tag_owned = tag.to_string();
        let stats = model.stats.clone();
        predict::run(move || async move {
            let start = Instant::now();
            let result = muna.predictions.create(
                &tag_owned,
                Some(inputs),
                Some(acceleration),
                None,
                None
            ).await;
            stats.record_latency(start.elapsed());
            result
        }).await
    }
}
