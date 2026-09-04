/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Per-model counters shared between the dispatcher and status reporting.
pub(crate) struct ModelStats {
    /// Total predictions completed over the model's lifetime.
    pub total_predictions: AtomicU64,
    /// Current number of items waiting in the dispatch queue.
    pub queue_depth: AtomicU32,
    /// Windowed timing samples for the heartbeat telemetry summary.
    pub telemetry: TelemetryWindow,
    /// Model load duration in microseconds: predictor creation only, with
    /// every resource already on disk.
    load_time_us: AtomicU64,
    /// Resource download duration in nanoseconds, measured before the load
    /// so `load_time_us` stays a pure cold-start number. Near zero (one
    /// manifest round trip) when the model was already disk-resident.
    download_time_ns: AtomicU64,
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
            telemetry: TelemetryWindow::new(),
            load_time_us: AtomicU64::new(0),
            download_time_ns: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            vram_mb: AtomicU64::new(0),
        }
    }

    /// Store the duration of the initial model load.
    pub(crate) fn record_load_time(&self, elapsed: std::time::Duration) {
        self.load_time_us.store(elapsed.as_micros() as u64, Ordering::Relaxed);
    }

    /// Store the duration of the resource download that preceded the load.
    pub(crate) fn record_download_time(&self, elapsed: std::time::Duration) {
        self.download_time_ns.store(elapsed.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Retrieve the download time in nanoseconds (near zero when nothing
    /// was fetched).
    pub(crate) fn download_time_ns(&self) -> u64 {
        self.download_time_ns.load(Ordering::Relaxed)
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

/// Timing sample for one completed prediction.
pub(crate) struct PredictionSample {
    /// When the sample was taken (completion time).
    pub at: Instant,
    /// Enqueue -> dispatch (the admission signal).
    pub queue_wait: Duration,
    /// Dispatch -> last output (total service time).
    pub latency: Duration,
    /// Per-surface decomposition.
    pub detail: SampleDetail,
}

/// Per-surface detail of a prediction sample. Only the handler that served
/// the request knows what its yields mean (an LLM chunk carries tokens, an
/// image is one unit, a raw stream's frames are opaque wire artifacts).
pub(crate) enum SampleDetail {
    /// Streaming OpenAI / Anthropic surfaces: token-normalized.
    Llm {
        /// Dispatch -> first content-bearing chunk (TTFT). Skips
        /// role-only / wire-consistency frames.
        first_output: Duration,
        /// (t_last - t_first) / tokens-after-first-chunk (TPOT), from
        /// the terminal usage chunk. None if the stream died early.
        output_interval: Option<Duration>,
    },
    /// images.generate: unary at the wire; one image per unit.
    Image {
        /// Number of images produced by the request.
        #[allow(dead_code)]
        images: u32,
    },
    /// Raw streaming prediction: yields are opaque, so only time to the
    /// first frame is kept.
    Stream {
        /// Dispatch -> first frame.
        first_output: Duration,
    },
    /// Unary prediction (raw, and non-streamed chat: a whole-response
    /// latency must not fatten the TTFT percentile the scaler watches).
    Unary,
}

/// Summary percentiles over a telemetry window. Every field is absent
/// unless enough samples of a variant that CARRIES that field exist -- an
/// image model simply never produces `output_interval` percentiles.
#[derive(Default, Clone)]
pub(crate) struct TelemetrySummary {
    /// p95 admission wait; header field, aggregates across all variants.
    pub queue_wait_ms_p95: Option<f64>,
    /// p95 total service time; header field, aggregates across all variants.
    pub latency_ms_p95: Option<f64>,
    /// p50 time to first output (`Llm` + `Stream` variants).
    pub first_output_ms_p50: Option<f64>,
    /// p95 time to first output (`Llm` + `Stream` variants).
    pub first_output_ms_p95: Option<f64>,
    /// p50 token-normalized output interval (`Llm` variant only).
    pub output_interval_ms_p50: Option<f64>,
    /// p95 token-normalized output interval (`Llm` variant only).
    pub output_interval_ms_p95: Option<f64>,
}

/// Sample cap: bounds memory per model; heartbeats read summaries, never
/// raw samples.
const WINDOW_CAP: usize = 512;
/// Only samples younger than this horizon contribute to a summary.
const WINDOW_HORIZON: Duration = Duration::from_secs(5 * 60);
/// Minimum contributing samples for any percentile to be reported.
const MIN_SAMPLES: usize = 5;

/// Fixed ring of recent samples; percentiles computed on read over a
/// 5-minute horizon.
pub(crate) struct TelemetryWindow {
    samples: Mutex<VecDeque<PredictionSample>>,
}

impl TelemetryWindow {

    pub(crate) fn new() -> Self {
        Self { samples: Mutex::new(VecDeque::with_capacity(WINDOW_CAP)) }
    }

    /// Append a sample, evicting the oldest past the ring capacity.
    pub(crate) fn record(&self, sample: PredictionSample) {
        let mut samples = self.samples.lock().unwrap();
        if samples.len() == WINDOW_CAP {
            samples.pop_front();
        }
        samples.push_back(sample);
    }

    /// p50/p95 over samples younger than 5 min; `None` with < 5 samples.
    pub(crate) fn summarize(&self) -> Option<TelemetrySummary> {
        let samples = self.samples.lock().unwrap();
        let now = Instant::now();
        let recent: Vec<&PredictionSample> = samples.iter()
            .filter(|sample| now.duration_since(sample.at) <= WINDOW_HORIZON)
            .collect();
        if recent.len() < MIN_SAMPLES {
            return None;
        }
        let queue_waits: Vec<f64> = recent.iter()
            .map(|sample| duration_ms(sample.queue_wait))
            .collect();
        let latencies: Vec<f64> = recent.iter()
            .map(|sample| duration_ms(sample.latency))
            .collect();
        let first_outputs: Vec<f64> = recent.iter()
            .filter_map(|sample| match &sample.detail {
                SampleDetail::Llm { first_output, .. } => Some(duration_ms(*first_output)),
                SampleDetail::Stream { first_output } => Some(duration_ms(*first_output)),
                _ => None,
            })
            .collect();
        let output_intervals: Vec<f64> = recent.iter()
            .filter_map(|sample| match &sample.detail {
                SampleDetail::Llm { output_interval: Some(interval), .. } => {
                    Some(duration_ms(*interval))
                }
                _ => None,
            })
            .collect();
        Some(TelemetrySummary {
            queue_wait_ms_p95: percentile(&queue_waits, 0.95),
            latency_ms_p95: percentile(&latencies, 0.95),
            first_output_ms_p50: percentile(&first_outputs, 0.50),
            first_output_ms_p95: percentile(&first_outputs, 0.95),
            output_interval_ms_p50: percentile(&output_intervals, 0.50),
            output_interval_ms_p95: percentile(&output_intervals, 0.95),
        })
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Nearest-rank percentile; `None` below the per-field sample floor.
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.len() < MIN_SAMPLES {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    Some(sorted[idx])
}

/// Which stream surface a [`StreamMeter`] is measuring; decides the
/// [`SampleDetail`] variant recorded at drop.
pub(crate) enum StreamKind {
    /// OpenAI / Anthropic chat stream: chunks carry tokens, the terminal
    /// usage frame carries the completion token count.
    Llm,
    /// Raw prediction stream: frames are opaque.
    Raw,
}

/// Live accumulator carried through a streaming response's `unfold` state;
/// records a [`PredictionSample`] when dropped (stream end or client
/// disconnect are the same hook). Streams that never produced output are
/// NOT recorded -- failure timings must not pollute quality percentiles.
pub(crate) struct StreamMeter {
    stats: Arc<ModelStats>,
    kind: StreamKind,
    queue_wait: Duration,
    started: Instant,
    first_content_at: Option<Instant>,
    last_output_at: Option<Instant>,
    content_chunks: u64,
    completion_tokens: Option<u64>,
}

impl StreamMeter {

    /// Start metering at dispatch time (call after the admission guard is
    /// acquired; `queue_wait` is the time spent acquiring it).
    pub(crate) fn new(
        stats: Arc<ModelStats>,
        kind: StreamKind,
        queue_wait: Duration
    ) -> Self {
        Self {
            stats,
            kind,
            queue_wait,
            started: Instant::now(),
            first_content_at: None,
            last_output_at: None,
            content_chunks: 0,
            completion_tokens: None,
        }
    }

    /// Stamp one stream frame. `content` marks a content-bearing frame
    /// (non-empty delta or usage -- not a role-only wire-consistency
    /// frame); only those anchor the first-output stamp.
    pub(crate) fn on_output(&mut self, content: bool) {
        let now = Instant::now();
        self.last_output_at = Some(now);
        if content {
            self.content_chunks += 1;
            if self.first_content_at.is_none() {
                self.first_content_at = Some(now);
            }
        }
    }

    /// Note the terminal usage frame's completion token count, enabling
    /// the yield-invariant output-interval normalization.
    pub(crate) fn on_usage(&mut self, completion_tokens: u64) {
        self.completion_tokens = Some(completion_tokens);
    }
}

impl Drop for StreamMeter {

    fn drop(&mut self) {
        let (Some(first), Some(last)) = (self.first_content_at, self.last_output_at) else {
            return;
        };
        let first_output = first.duration_since(self.started);
        let detail = match self.kind {
            StreamKind::Raw => SampleDetail::Stream { first_output },
            StreamKind::Llm => {
                let output_interval = self.completion_tokens.and_then(|tokens| {
                    normalize_interval(
                        last.duration_since(first),
                        tokens,
                        self.content_chunks
                    )
                });
                SampleDetail::Llm { first_output, output_interval }
            }
        };
        self.stats.telemetry.record(PredictionSample {
            at: Instant::now(),
            queue_wait: self.queue_wait,
            latency: last.duration_since(self.started),
            detail,
        });
    }
}

/// Yield-invariant TPOT: chunk cadence tracks engine steps (a speculative
/// decoder commits several tokens per yield), so normalize the first->last
/// span by the tokens delivered AFTER the first chunk. The first chunk's
/// token share is not on the wire; estimate it as the mean tokens-per-chunk.
fn normalize_interval(
    span: Duration,
    completion_tokens: u64,
    content_chunks: u64
) -> Option<Duration> {
    if content_chunks < 2 || completion_tokens == 0 {
        return None;
    }
    let per_chunk = completion_tokens.div_ceil(content_chunks);
    let after_first = completion_tokens.saturating_sub(per_chunk);
    if after_first == 0 {
        return None;
    }
    Some(span / after_first as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(age: Duration, queue_wait_ms: u64, latency_ms: u64) -> PredictionSample {
        PredictionSample {
            at: Instant::now().checked_sub(age).expect("test age fits"),
            queue_wait: Duration::from_millis(queue_wait_ms),
            latency: Duration::from_millis(latency_ms),
            detail: SampleDetail::Unary,
        }
    }

    #[test]
    fn summarize_requires_min_samples() {
        let window = TelemetryWindow::new();
        for _ in 0..(MIN_SAMPLES - 1) {
            window.record(sample(Duration::ZERO, 1, 10));
        }
        assert!(window.summarize().is_none());
        window.record(sample(Duration::ZERO, 1, 10));
        assert!(window.summarize().is_some());
    }

    #[test]
    fn summarize_ignores_samples_past_horizon() {
        let window = TelemetryWindow::new();
        // Ten stale samples with huge latency, five fresh ones with small.
        for _ in 0..10 {
            window.record(sample(WINDOW_HORIZON + Duration::from_secs(1), 1, 10_000));
        }
        for _ in 0..5 {
            window.record(sample(Duration::ZERO, 1, 10));
        }
        let summary = window.summarize().expect("five fresh samples");
        assert!(summary.latency_ms_p95.expect("header field") < 100.0);
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let window = TelemetryWindow::new();
        for ms in 1..=100u64 {
            window.record(sample(Duration::ZERO, ms, ms));
        }
        let summary = window.summarize().expect("100 samples");
        let p95 = summary.latency_ms_p95.expect("header field");
        assert!((94.0..=96.0).contains(&p95), "p95 = {p95}");
    }

    #[test]
    fn ring_caps_at_window_size() {
        let window = TelemetryWindow::new();
        for _ in 0..(WINDOW_CAP + 100) {
            window.record(sample(Duration::ZERO, 1, 10));
        }
        assert_eq!(window.samples.lock().unwrap().len(), WINDOW_CAP);
    }

    #[test]
    fn variant_fields_project_per_variant() {
        let window = TelemetryWindow::new();
        // Image samples never produce first_output / output_interval fields.
        for _ in 0..10 {
            window.record(PredictionSample {
                at: Instant::now(),
                queue_wait: Duration::from_millis(1),
                latency: Duration::from_millis(500),
                detail: SampleDetail::Image { images: 1 },
            });
        }
        let summary = window.summarize().expect("ten samples");
        assert!(summary.queue_wait_ms_p95.is_some());
        assert!(summary.latency_ms_p95.is_some());
        assert!(summary.first_output_ms_p50.is_none());
        assert!(summary.output_interval_ms_p50.is_none());
        // Llm samples fill the streaming fields once they cross the floor.
        for _ in 0..MIN_SAMPLES {
            window.record(PredictionSample {
                at: Instant::now(),
                queue_wait: Duration::from_millis(1),
                latency: Duration::from_millis(500),
                detail: SampleDetail::Llm {
                    first_output: Duration::from_millis(50),
                    output_interval: Some(Duration::from_millis(8)),
                },
            });
        }
        let summary = window.summarize().expect("fifteen samples");
        assert!(summary.first_output_ms_p50.is_some());
        assert!(summary.first_output_ms_p95.is_some());
        assert!(summary.output_interval_ms_p50.is_some());
        assert!(summary.output_interval_ms_p95.is_some());
    }

    #[test]
    fn interval_normalization_is_yield_invariant() {
        // 100 tokens over 990ms after the first chunk. Delivered as 100
        // one-token chunks (autoregressive) or 25 four-token chunks
        // (speculative), the per-token interval must match: ~10ms.
        let span = Duration::from_millis(990);
        let autoregressive = normalize_interval(span, 100, 100).expect("interval");
        let speculative = normalize_interval(span, 100, 25).expect("interval");
        assert_eq!(autoregressive, Duration::from_millis(10));
        // Speculative: per_chunk = 4, after_first = 96 -> 990/96 ~= 10.3ms.
        let ms = speculative.as_secs_f64() * 1000.0;
        assert!((9.0..=11.5).contains(&ms), "speculative interval = {ms}ms");
    }

    #[test]
    fn interval_normalization_degenerate_cases() {
        let span = Duration::from_millis(100);
        // Single chunk: no span to normalize.
        assert!(normalize_interval(span, 10, 1).is_none());
        // No tokens reported.
        assert!(normalize_interval(span, 0, 10).is_none());
        // All tokens attributed to the first chunk estimate.
        assert!(normalize_interval(span, 1, 2).is_none());
    }
}
