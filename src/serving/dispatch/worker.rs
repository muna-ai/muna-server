/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Buffered accumulator: one task per buffered model that files incoming
//! requests into per-batch-key queues, flushes whichever batch fills or
//! expires first (earliest deadline wins), invokes the model once per batch,
//! and splits the results back per caller.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use muna::types::{Acceleration, Prediction, Value};
use muna::MunaError;
use tokio::time::timeout_at;

use super::merge::{merge_inputs, split_results};
use crate::serving::stats::{ModelStats, PredictionSample, SampleDetail};

/// How long the accumulator waits for more requests before flushing.
const FLUSH_DEADLINE: Duration = Duration::from_millis(100);

/// Capacity of each buffered model's dispatch channel.
pub(super) const CHANNEL_BUFFER: usize = 1024;

/// Prediction delegate for the buffered worker. Injectable so flush
/// behavior is testable without a live engine.
pub(super) type PredictFn = Arc<
    dyn Fn(
            HashMap<String, Value>,
            Acceleration
        ) -> futures_util::future::BoxFuture<'static, Result<Prediction, MunaError>>
        + Send
        + Sync
>;

/// One prediction request in flight through a buffered model's accumulator.
///
/// Created by `Dispatcher::create` and sent over the model's channel to its
/// `BufferedWorker`, which merges compatible items into a batch, invokes the
/// model once, and answers each caller through its `response_tx`.
pub(super) struct PredictItem {
    /// Raw prediction inputs as supplied by the caller.
    pub(super) inputs: HashMap<String, Value>,
    /// Acceleration the caller requested for the invocation.
    pub(super) acceleration: Acceleration,
    /// How many batch slots this request occupies: the maximum list length
    /// among its batch parameters (broadcast-only requests count as one).
    pub(super) item_count: usize,
    /// Merge-compatibility key: a deterministic fingerprint of the request's
    /// broadcast (non-batched) values. Only items with equal keys may share
    /// an invocation, so broadcast values are never silently overwritten.
    pub(super) batch_key: String,
    /// When the caller dispatched the request. Anchors the flush deadline,
    /// so time spent in the channel (e.g. during an invocation) counts.
    pub(super) enqueued: tokio::time::Instant,
    /// Oneshot channel over which the worker delivers this request's slice
    /// of the batched prediction (or the whole-batch error).
    pub(super) response_tx: tokio::sync::oneshot::Sender<Result<Prediction, MunaError>>,
}

/// FIFO of same-key requests awaiting a batch slot. The front item's
/// arrival time anchors the flush deadline for the whole queue.
struct KeyQueue {
    items: VecDeque<PredictItem>,
    total_items: usize,
}

impl KeyQueue {

    fn deadline(&self) -> tokio::time::Instant {
        let front = self.items.front().expect("KeyQueue is never empty");
        front.enqueued + FLUSH_DEADLINE
    }

    /// Take the largest prefix whose item count fits in `capacity` (always
    /// at least one item: a single oversized request runs alone).
    fn take_batch(&mut self, capacity: usize) -> Vec<PredictItem> {
        let mut batch = Vec::new();
        let mut total = 0usize;
        while let Some(front) = self.items.front() {
            if !batch.is_empty() && total + front.item_count > capacity {
                break;
            }
            total += front.item_count;
            let item = self.items.pop_front().expect("front just observed");
            self.total_items -= item.item_count;
            batch.push(item);
            if total >= capacity {
                break;
            }
        }
        batch
    }
}

/// File one request into its key's queue, creating the queue on first use.
fn file_item(
    queues: &mut HashMap<String, KeyQueue>,
    item: PredictItem
) {
    let queue = queues
        .entry(item.batch_key.clone())
        .or_insert_with(|| KeyQueue { items: VecDeque::new(), total_items: 0 });
    queue.total_items += item.item_count;
    queue.items.push_back(item);
}

/// Pop the next batch to flush: among keys at capacity or past their
/// deadline, earliest deadline first (arrival order breaks ties, so no key
/// starves; full batches become eligible immediately). Emptied queues are
/// removed -- batch keys have unbounded cardinality, so stale entries must
/// not accumulate.
fn take_flushable(
    queues: &mut HashMap<String, KeyQueue>,
    capacity: usize
) -> Option<Vec<PredictItem>> {
    let now = tokio::time::Instant::now();
    let key = queues.iter()
        .filter(|(_, queue)| {
            queue.total_items >= capacity || queue.deadline() <= now
        })
        .min_by_key(|(_, queue)| queue.deadline())
        .map(|(key, _)| key.clone())?;
    let queue = queues.get_mut(&key).expect("key just selected");
    let batch = queue.take_batch(capacity);
    if queue.items.is_empty() {
        queues.remove(&key);
    }
    Some(batch)
}

fn earliest_deadline(
    queues: &HashMap<String, KeyQueue>
) -> Option<tokio::time::Instant> {
    queues.values().map(KeyQueue::deadline).min()
}

/// Accumulator task for one buffered model.
pub(super) struct BufferedWorker {
    pub(super) stats: Arc<ModelStats>,
    pub(super) params: HashSet<String>,
    pub(super) capacity: usize,
    pub(super) predict_fn: PredictFn,
    pub(super) rx: async_channel::Receiver<PredictItem>,
}

impl BufferedWorker {

    pub(super) async fn run(self) {
        // Open batches accumulate per batch key, each with a deadline
        // anchored to its oldest request's arrival. A mismatched-key
        // arrival therefore never blocks (or is blocked by) another key's
        // batch, and a deferred request's clock never restarts.
        let mut queues: HashMap<String, KeyQueue> = HashMap::new();
        loop {
            // File everything already waiting (includes arrivals that
            // landed during the previous invocation).
            while let Ok(item) = self.rx.try_recv() {
                self.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                file_item(&mut queues, item);
            }
            // Static and dynamic modes share this flush behavior by design:
            // partial batches are invoked as-is, never padded. The server
            // cannot know a valid pad value for opaque inputs, so a
            // predictor with a rigid batch shape must pad internally.
            if let Some(batch) = take_flushable(&mut queues, self.capacity) {
                // Invocations are serialized by construction: buffered mode
                // exists because the model does not tolerate concurrent
                // invocation. Keying only changes which batch runs next.
                self.flush(batch).await;
                continue;
            }
            // Nothing eligible: wait for the next arrival or the earliest
            // open-queue deadline, whichever comes first.
            match earliest_deadline(&queues) {
                Some(deadline) => match timeout_at(deadline, self.rx.recv()).await {
                    Ok(Ok(item)) => {
                        self.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        file_item(&mut queues, item);
                    }
                    // All senders dropped: the model was unloaded.
                    Ok(Err(_)) => break,
                    // Deadline expired: the next iteration flushes it.
                    Err(_) => {}
                },
                None => match self.rx.recv().await {
                    Ok(item) => {
                        self.stats.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        file_item(&mut queues, item);
                    }
                    Err(_) => break,
                },
            }
        }
    }

    async fn flush(&self, batch: Vec<PredictItem>) {
        // Queue wait is per request (enqueue -> dispatch); latency is the
        // shared invocation's elapsed time, which is each request's service
        // time -- requests merged into a batch complete together.
        let dispatched = tokio::time::Instant::now();
        let queue_waits: Vec<Duration> = batch.iter()
            .map(|item| dispatched.duration_since(item.enqueued))
            .collect();
        if batch.len() == 1 {
            let item = batch.into_iter().next().expect("batch of one");
            let result = (self.predict_fn)(item.inputs, item.acceleration).await;
            let _ = item.response_tx.send(result);
        } else {
            self.process_batch(batch).await;
        }
        let latency = dispatched.elapsed();
        for queue_wait in queue_waits {
            self.stats.telemetry.record(PredictionSample {
                at: std::time::Instant::now(),
                queue_wait,
                latency,
                detail: SampleDetail::Unary,
            });
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

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
            enqueued: tokio::time::Instant::now(),
            response_tx: tx,
        };
        (item, rx)
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
        calls: Arc<Mutex<Vec<usize>>>
    ) -> async_channel::Sender<PredictItem> {
        let (tx, rx) = async_channel::bounded(CHANNEL_BUFFER);
        let worker = BufferedWorker {
            stats: Arc::new(ModelStats::new()),
            params: HashSet::from(["texts".into()]),
            capacity,
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
        let tx = spawn_worker(8, calls.clone());
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
        let tx = spawn_worker(3, calls.clone());
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
        let tx = spawn_worker(8, calls.clone());
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
        let tx = spawn_worker(8, calls.clone());
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

    #[tokio::test]
    async fn buffered_interleaved_mismatch_does_not_split_batch() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(8, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a"]), 1, "dims=10");
        let (c, c_rx) = make_item(text_inputs(&["c"]), 1, "dims=20");
        let (b, b_rx) = make_item(text_inputs(&["b"]), 1, "dims=10");
        // The mismatched key arrives BETWEEN two same-key requests: it must
        // be deferred to its own batch, not end the current accumulation.
        tx.send(a).await.unwrap();
        tx.send(c).await.unwrap();
        tx.send(b).await.unwrap();
        let a_result = a_rx.await.unwrap().unwrap();
        let b_result = b_rx.await.unwrap().unwrap();
        let c_result = c_rx.await.unwrap().unwrap();
        // Same-key requests merge into one 2-item invocation; the
        // mismatched key gets its own 1-item invocation afterwards.
        assert_eq!(*calls.lock().unwrap(), vec![2, 1]);
        assert!(a_result.error.is_none());
        assert!(b_result.error.is_none());
        assert!(c_result.error.is_none());
    }

    #[tokio::test]
    async fn full_batch_preempts_older_partial_batch() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(2, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a"]), 1, "dims=10");
        let (b, b_rx) = make_item(text_inputs(&["b"]), 1, "dims=20");
        let (c, c_rx) = make_item(text_inputs(&["c"]), 1, "dims=20");
        // Key dims=10 arrives first but stays partial; dims=20 fills to
        // capacity and must flush FIRST, without waiting out dims=10's
        // deadline.
        tx.send(a).await.unwrap();
        tx.send(b).await.unwrap();
        tx.send(c).await.unwrap();
        let b_result = tokio::time::timeout(Duration::from_millis(90), b_rx)
            .await
            .expect("full batch must flush before any deadline elapses")
            .unwrap()
            .unwrap();
        let a_result = a_rx.await.unwrap().unwrap();
        let c_result = c_rx.await.unwrap().unwrap();
        // The 2-item invocation is the full dims=20 batch; the older
        // partial dims=10 batch follows at its deadline.
        assert_eq!(*calls.lock().unwrap(), vec![2, 1]);
        assert!(a_result.error.is_none());
        assert!(b_result.error.is_none());
        assert!(c_result.error.is_none());
    }

    #[tokio::test]
    async fn deferred_key_deadline_anchored_at_arrival() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(8, calls.clone());
        let (a, a_rx) = make_item(text_inputs(&["a"]), 1, "dims=10");
        let (b, b_rx) = make_item(text_inputs(&["b"]), 1, "dims=20");
        let start = std::time::Instant::now();
        tx.send(a).await.unwrap();
        tx.send(b).await.unwrap();
        let a_result = a_rx.await.unwrap().unwrap();
        let b_result = b_rx.await.unwrap().unwrap();
        // b's deadline is anchored at ITS arrival, not restarted after a's
        // batch flushed: both flush ~one FLUSH_DEADLINE after send, where
        // the old single-batch worker took ~two for the deferred key.
        assert!(
            start.elapsed() < FLUSH_DEADLINE + FLUSH_DEADLINE / 2,
            "deferred key waited {:?}; its clock must not restart",
            start.elapsed()
        );
        assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
        assert!(a_result.error.is_none());
        assert!(b_result.error.is_none());
    }

    #[tokio::test]
    async fn oversized_request_runs_alone() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let tx = spawn_worker(2, calls.clone());
        let (a, a_rx) = make_item(
            text_inputs(&["a", "b", "c", "d", "e"]),
            5,
            ""
        );
        tx.send(a).await.unwrap();
        // Item count exceeds capacity: the request is immediately eligible
        // and runs alone (a single request can never be split).
        let result = tokio::time::timeout(Duration::from_millis(90), a_rx)
            .await
            .expect("oversized request must not wait for the deadline")
            .unwrap()
            .unwrap();
        assert_eq!(*calls.lock().unwrap(), vec![5]);
        assert!(matches!(
            &result.results.unwrap()[0],
            Value::List(items) if items.len() == 5
        ));
    }
}
