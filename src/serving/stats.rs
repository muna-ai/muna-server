/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Per-model counters shared between the dispatcher and status reporting.
pub(crate) struct ModelStats {
    /// Total predictions completed over the model's lifetime.
    pub total_predictions: AtomicU64,
    /// Current number of items waiting in the dispatch queue.
    pub queue_depth: AtomicU32,
    /// Model load duration in microseconds.
    load_time_us: AtomicU64,
    /// Running sum of per-prediction latencies in microseconds.
    latency_sum_us: AtomicU64,
    /// Number of latency samples recorded (denominator for the average).
    latency_count: AtomicU64,
    /// Estimated memory used by this model in MB (measured as a delta at load time).
    vram_mb: AtomicU64,
}

impl ModelStats {

    pub(crate) fn new() -> Self {
        Self {
            total_predictions: AtomicU64::new(0),
            queue_depth: AtomicU32::new(0),
            load_time_us: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            vram_mb: AtomicU64::new(0),
        }
    }

    /// Store the duration of the initial model load.
    pub(crate) fn record_load_time(&self, elapsed: std::time::Duration) {
        self.load_time_us.store(elapsed.as_micros() as u64, Ordering::Relaxed);
    }

    /// Accumulate a single prediction's latency and bump the counter.
    pub(crate) fn record_latency(&self, elapsed: std::time::Duration) {
        self.latency_sum_us.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Retrieve the load time in milliseconds.
    pub(crate) fn load_time_ms(&self) -> f64 {
        self.load_time_us.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Retrieve the mean prediction latency in milliseconds.
    pub(crate) fn avg_latency_ms(&self) -> f64 {
        let count = self.latency_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        self.latency_sum_us.load(Ordering::Relaxed) as f64 / count as f64 / 1000.0
    }

    /// Store the memory delta measured at model load time.
    pub(crate) fn record_vram(&self, mb: u64) {
        self.vram_mb.store(mb, Ordering::Relaxed);
    }

    /// Retrieve the estimated memory usage in MB, or `None` if not yet measured.
    pub(crate) fn vram_mb(&self) -> Option<u64> {
        let v = self.vram_mb.load(Ordering::Relaxed);
        if v > 0 { Some(v) } else { None }
    }
}
