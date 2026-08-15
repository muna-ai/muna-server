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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Prediction, Tensor, TensorData, Value};
use muna::{Muna, MunaError};
use tokio::time::timeout_at;

use crate::serving::batch::{compute_batch_key, item_count, BatchPlan};
use crate::serving::predict;
use crate::serving::registry::ReadyModel;
use crate::serving::stats::ModelStats;

/// How long the accumulator waits for more requests before flushing.
const FLUSH_DEADLINE: Duration = Duration::from_millis(100);

const CHANNEL_BUFFER: usize = 1024;

struct PredictItem {
    inputs: HashMap<String, Value>,
    acceleration: Acceleration,
    item_count: usize,
    batch_key: String,
    response_tx: tokio::sync::oneshot::Sender<Result<Prediction, MunaError>>,
}

enum Entry {
    Sequential { lock: Arc<tokio::sync::Mutex<()>> },
    Buffered { tx: async_channel::Sender<PredictItem> },
    Continuous,
}

#[derive(Clone)]
pub(crate) struct Dispatcher {
    muna: Arc<Muna>,
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
            BatchPlan::Buffered { params, capacity, wait_full } => {
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
                    wait_full: *wait_full,
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

/// Prediction delegate for the buffered worker. Injectable so flush
/// behavior is testable without a live engine.
type PredictFn = Arc<
    dyn Fn(
            HashMap<String, Value>,
            Acceleration
        ) -> futures_util::future::BoxFuture<'static, Result<Prediction, MunaError>>
        + Send
        + Sync
>;

/// Accumulator task for one buffered model.
struct BufferedWorker {
    stats: Arc<ModelStats>,
    params: HashSet<String>,
    capacity: usize,
    wait_full: bool,
    predict_fn: PredictFn,
    rx: async_channel::Receiver<PredictItem>,
}

impl BufferedWorker {

    async fn run(self) {
        let mut held: Option<PredictItem> = None;
        loop {
            let first = match held.take() {
                Some(item) => item,
                None => match self.rx.recv().await {
                    Ok(item) => {
                        self.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        item
                    }
                    // All senders dropped: the model was unloaded.
                    Err(_) => break,
                },
            };
            let first_key = first.batch_key.clone();
            let mut total_items = first.item_count;
            let mut batch = vec![first];
            // Static (`wait_full`) and dynamic currently share the flush
            // behavior: accumulate until capacity or deadline. They diverge
            // once padding lands (static pads a partial batch to capacity).
            let _ = self.wait_full;
            let deadline = tokio::time::Instant::now() + FLUSH_DEADLINE;
            while total_items < self.capacity {
                match timeout_at(deadline, self.rx.recv()).await {
                    Ok(Ok(item)) => {
                        self.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        if item.batch_key != first_key ||
                            total_items + item.item_count > self.capacity {
                            // Mismatched key or overflow: hold for next batch.
                            held = Some(item);
                            break;
                        }
                        total_items += item.item_count;
                        batch.push(item);
                    }
                    _ => break,
                }
            }
            if batch.len() == 1 {
                let item = batch.into_iter().next().unwrap();
                let result = (self.predict_fn)(item.inputs, item.acceleration).await;
                let _ = item.response_tx.send(result);
            } else {
                self.process_batch(batch).await;
            }
        }
    }

    async fn process_batch(&self, batch: Vec<PredictItem>) {
        let counts: Vec<usize> = batch.iter().map(|b| b.item_count).collect();
        let inputs: Vec<&HashMap<String, Value>> = batch.iter().map(|b| &b.inputs).collect();
        let merged = merge_inputs(&inputs, &self.params);
        // Acceleration is not part of the batch key; the first request's
        // choice applies to the merged invocation.
        let acceleration = batch[0].acceleration.clone();
        match (self.predict_fn)(merged, acceleration).await {
            Ok(prediction) => {
                let results = prediction.results.clone().unwrap_or_default();
                let splits = split_results(results, &counts);
                for (item, split) in batch.into_iter().zip(splits) {
                    let p = Prediction {
                        results: Some(split),
                        ..prediction.clone()
                    };
                    let _ = item.response_tx.send(Ok(p));
                }
            }
            Err(e) => {
                for item in batch {
                    let _ = item.response_tx.send(Err(MunaError::Native(e.to_string())));
                }
            }
        }
    }
}

/// Merge a batch of compatible requests into one input map: batch params are
/// concatenated in request order; broadcast params come from the first
/// request (identical across the batch by batch-key construction).
pub(crate) fn merge_inputs(
    batch: &[&HashMap<String, Value>],
    batch_params: &HashSet<String>,
) -> HashMap<String, Value> {
    if batch.len() == 1 {
        return batch[0].clone();
    }
    let mut merged = HashMap::new();
    for (key, value) in batch[0] {
        if !batch_params.contains(key.as_str()) {
            merged.insert(key.clone(), value.clone());
        }
    }
    for name in batch_params {
        let mut combined = Vec::new();
        for inputs in batch {
            if let Some(Value::List(items)) = inputs.get(name) {
                combined.extend(items.iter().cloned());
            }
        }
        merged.insert(name.clone(), Value::List(combined));
    }
    merged
}

/// Split merged prediction results back per request: tensors split on dim 0
/// by item counts, lists by counts, everything else broadcasts.
pub(crate) fn split_results(
    results: Vec<Value>,
    counts: &[usize],
) -> Vec<Vec<Value>> {
    let total: usize = counts.iter().sum();
    let n = counts.len();
    let mut splits: Vec<Vec<Value>> = (0..n).map(|_| Vec::new()).collect();
    for value in results {
        match &value {
            Value::Tensor(tensor)
                if !tensor.shape.is_empty() && tensor.shape[0] as usize == total => {
                let inner: usize = tensor.shape[1..].iter()
                    .map(|&s| s as usize)
                    .product::<usize>()
                    .max(1);
                let mut offset = 0;
                for (i, &count) in counts.iter().enumerate() {
                    let start = offset * inner;
                    let end = (offset + count) * inner;
                    let slice_shape = {
                        let mut s = tensor.shape.clone();
                        s[0] = count as i32;
                        s
                    };
                    splits[i].push(Value::Tensor(Tensor {
                        data: slice_tensor_data(&tensor.data, start, end),
                        shape: slice_shape,
                    }));
                    offset += count;
                }
            }
            Value::List(items) if items.len() == total => {
                let mut offset = 0;
                for (i, &count) in counts.iter().enumerate() {
                    splits[i].push(Value::List(items[offset..offset + count].to_vec()));
                    offset += count;
                }
            }
            other => {
                for s in &mut splits {
                    s.push(other.clone());
                }
            }
        }
    }
    splits
}

fn slice_tensor_data(data: &TensorData, start: usize, end: usize) -> TensorData {
    match data {
        TensorData::Float32(v) => TensorData::Float32(v[start..end].to_vec()),
        TensorData::Float64(v) => TensorData::Float64(v[start..end].to_vec()),
        TensorData::Int8(v) => TensorData::Int8(v[start..end].to_vec()),
        TensorData::Int16(v) => TensorData::Int16(v[start..end].to_vec()),
        TensorData::Int32(v) => TensorData::Int32(v[start..end].to_vec()),
        TensorData::Int64(v) => TensorData::Int64(v[start..end].to_vec()),
        TensorData::Uint8(v) => TensorData::Uint8(v[start..end].to_vec()),
        TensorData::Uint16(v) => TensorData::Uint16(v[start..end].to_vec()),
        TensorData::Uint32(v) => TensorData::Uint32(v[start..end].to_vec()),
        TensorData::Uint64(v) => TensorData::Uint64(v[start..end].to_vec()),
        TensorData::Complex64(v) => TensorData::Complex64(v[start..end].to_vec()),
        TensorData::Complex128(v) => TensorData::Complex128(v[start..end].to_vec()),
        TensorData::Bool(v) => TensorData::Bool(v[start..end].to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;

    fn make_item(
        inputs: HashMap<String, Value>,
        item_count: usize,
        batch_key: &str,
    ) -> (PredictItem, tokio::sync::oneshot::Receiver<Result<Prediction, MunaError>>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let item = PredictItem {
            inputs,
            acceleration: Acceleration::LocalGpu,
            item_count,
            batch_key: batch_key.into(),
            response_tx: tx,
        };
        (item, rx)
    }

    #[test]
    fn merge_single_item_returns_inputs() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let inputs = HashMap::from([
            ("texts".into(), Value::List(vec!["hello".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let merged = merge_inputs(&[&inputs], &bp);
        assert_eq!(merged.len(), 2);
        assert!(matches!(merged.get("dims"), Some(Value::Int(10))));
    }

    #[test]
    fn merge_concatenates_batch_params() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec!["a".into(), "b".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec!["c".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let merged = merge_inputs(&[&a, &b], &bp);
        match merged.get("texts") {
            Some(Value::List(items)) => assert_eq!(items.len(), 3),
            other => panic!("expected list of 3, got {other:?}"),
        }
        assert!(matches!(merged.get("dims"), Some(Value::Int(10))));
    }

    #[test]
    fn split_tensor_along_dim0() {
        let tensor = Value::Tensor(Tensor {
            data: TensorData::Float32(vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ]),
            shape: vec![3, 4],
        });
        let splits = split_results(vec![tensor], &[2, 1]);
        assert_eq!(splits.len(), 2);
        match &splits[0][0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert!(matches!(
                    &t.data,
                    TensorData::Float32(v) if v == &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                ));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match &splits[1][0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(matches!(
                    &t.data,
                    TensorData::Float32(v) if v == &[9.0, 10.0, 11.0, 12.0]
                ));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn split_list_by_counts() {
        let list = Value::List(vec![
            "a".into(), "b".into(), "c".into(), "d".into(), "e".into()
        ]);
        let splits = split_results(vec![list], &[3, 2]);
        assert_eq!(splits.len(), 2);
        match &splits[0][0] {
            Value::List(items) => assert_eq!(items.len(), 3),
            other => panic!("expected list, got {other:?}"),
        }
        match &splits[1][0] {
            Value::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn split_scalar_broadcasts() {
        let splits = split_results(vec![Value::Float(42.0)], &[2, 3]);
        assert!(matches!(&splits[0][0], Value::Float(f) if *f == 42.0));
        assert!(matches!(&splits[1][0], Value::Float(f) if *f == 42.0));
    }

    #[test]
    fn split_tensor_dim0_mismatch_broadcasts() {
        let tensor = Value::Tensor(Tensor {
            data: TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![4, 1],
        });
        let splits = split_results(vec![tensor], &[2, 1]);
        match (&splits[0][0], &splits[1][0]) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(a.shape, vec![4, 1]);
                assert_eq!(b.shape, vec![4, 1]);
            }
            _ => panic!("expected tensors"),
        }
    }

    #[test]
    fn merge_split_round_trip() {
        // Two requests of 2 + 1 items merge into a 3-item invocation whose
        // list output splits back to the original request shapes.
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".to_string(), Value::List(vec!["a".into(), "b".into()])),
        ]);
        let b = HashMap::from([
            ("texts".to_string(), Value::List(vec!["c".into()])),
        ]);
        let merged = merge_inputs(&[&a, &b], &bp);
        let merged_texts = match merged.get("texts") {
            Some(Value::List(items)) => items.clone(),
            other => panic!("expected list, got {other:?}"),
        };
        let splits = split_results(vec![Value::List(merged_texts)], &[2, 1]);
        assert!(matches!(&splits[0][0], Value::List(items) if items.len() == 2));
        assert!(matches!(&splits[1][0], Value::List(items) if items.len() == 1));
    }

    /// Fake predictor that records each invocation's merged batch-param list
    /// length and echoes the list back as the sole result.
    fn recording_predictor(
        calls: Arc<Mutex<Vec<usize>>>
    ) -> PredictFn {
        Arc::new(move |inputs, _acceleration| {
            let calls = calls.clone();
            Box::pin(async move {
                let count = match inputs.get("texts") {
                    Some(Value::List(items)) => items.len(),
                    _ => 0,
                };
                calls.lock().unwrap().push(count);
                let results = inputs.get("texts").cloned().map(|v| vec![v]);
                Ok(Prediction {
                    id: "pred_test".into(),
                    tag: "@test/model".into(),
                    created: "0".into(),
                    configuration: None,
                    resources: None,
                    results,
                    latency: None,
                    error: None,
                    logs: None,
                })
            })
        })
    }

    fn spawn_worker(
        capacity: usize,
        wait_full: bool,
        calls: Arc<Mutex<Vec<usize>>>
    ) -> async_channel::Sender<PredictItem> {
        let (tx, rx) = async_channel::bounded(CHANNEL_BUFFER);
        let worker = BufferedWorker {
            stats: Arc::new(ModelStats::new()),
            params: HashSet::from(["texts".into()]),
            capacity,
            wait_full,
            predict_fn: recording_predictor(calls),
            rx,
        };
        tokio::spawn(worker.run());
        tx
    }

    fn text_inputs(texts: &[&str]) -> HashMap<String, Value> {
        HashMap::from([(
            "texts".to_string(),
            Value::List(texts.iter().map(|t| (*t).into()).collect())
        )])
    }

    #[tokio::test]
    async fn buffered_merges_within_deadline() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(8, false, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a", "b"]), 2, "");
        let (b, b_rx) = make_item(text_inputs(&["c"]), 1, "");
        tx.send(a).await.unwrap();
        tx.send(b).await.unwrap();
        let a_result = a_rx.await.unwrap().unwrap();
        let b_result = b_rx.await.unwrap().unwrap();
        // One merged invocation of 3 items, split back 2 + 1.
        assert_eq!(*calls.lock().unwrap(), vec![3]);
        assert!(matches!(
            &a_result.results.unwrap()[0],
            Value::List(items) if items.len() == 2
        ));
        assert!(matches!(
            &b_result.results.unwrap()[0],
            Value::List(items) if items.len() == 1
        ));
    }

    #[tokio::test]
    async fn buffered_flushes_at_capacity() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(3, true, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a", "b"]), 2, "");
        let (b, b_rx) = make_item(text_inputs(&["c"]), 1, "");
        tx.send(a).await.unwrap();
        tx.send(b).await.unwrap();
        // Capacity reached: the flush happens without waiting out the
        // deadline. Bound the wait well below FLUSH_DEADLINE margin.
        let a_result = tokio::time::timeout(Duration::from_millis(90), a_rx)
            .await
            .expect("capacity flush should not wait for the deadline")
            .unwrap()
            .unwrap();
        let b_result = b_rx.await.unwrap().unwrap();
        assert_eq!(*calls.lock().unwrap(), vec![3]);
        assert!(a_result.error.is_none());
        assert!(b_result.error.is_none());
    }

    #[tokio::test]
    async fn buffered_partial_batch_flushes_at_deadline() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(8, false, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a"]), 1, "");
        let start = std::time::Instant::now();
        tx.send(a).await.unwrap();
        let result = a_rx.await.unwrap().unwrap();
        // Dynamic mode: the lone item flushes once the deadline elapses.
        assert!(start.elapsed() >= FLUSH_DEADLINE);
        assert_eq!(*calls.lock().unwrap(), vec![1]);
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn buffered_mismatched_key_held_for_next_batch() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(8, false, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a"]), 1, "dims=10");
        let (b, b_rx) = make_item(text_inputs(&["b"]), 1, "dims=20");
        tx.send(a).await.unwrap();
        tx.send(b).await.unwrap();
        let a_result = a_rx.await.unwrap().unwrap();
        let b_result = b_rx.await.unwrap().unwrap();
        // Two invocations of one item each; the keys never merge.
        assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
        assert!(a_result.error.is_none());
        assert!(b_result.error.is_none());
    }
}
