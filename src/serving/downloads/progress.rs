/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Terminal and structured-log presentation for a physical resource transfer.

use std::io::IsTerminal;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Shared draw surface so concurrent downloads render stacked bars instead
/// of interleaving redraws.
fn multi_progress() -> &'static MultiProgress {
    static MULTI: OnceLock<MultiProgress> = OnceLock::new();
    MULTI.get_or_init(MultiProgress::new)
}

/// How often the non-TTY fallback emits a progress log line.
const PROGRESS_LOG_INTERVAL_MS: u64 = 5_000;

/// Per-file download progress presentation, fed by the muna client's
/// progress callback. On a TTY this renders an indicatif bar (muna-py
/// parity: its `client.download` draws a `rich.progress` bar by default);
/// indicatif hides itself when stderr is not a terminal, so in containers/CI
/// a throttled `tracing` line is emitted instead. The bar is the single
/// source of truth for position/elapsed/rate (indicatif tracks them even
/// when hidden); only the log throttle is ours.
#[derive(Clone)]
pub(super) struct DownloadProgress {
    name: String,
    bar: ProgressBar,
    /// Bar-elapsed milliseconds of the last emitted log line.
    last_log_ms: Arc<AtomicU64>,
    /// Emit tracing lines instead of the (hidden) bar.
    log_fallback: bool,
}

impl DownloadProgress {
    pub(super) fn new(path: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("resource")
            .to_string();
        // Total size arrives with the first callback (from the client's
        // range probe); start as a spinner and grow a bar if it is known.
        let bar = ProgressBar::no_length().with_style(spinner_style());
        let bar = multi_progress().add(bar);
        bar.set_message(name.clone());
        Self {
            name,
            bar,
            last_log_ms: Arc::new(AtomicU64::new(0)),
            log_fallback: !std::io::stderr().is_terminal(),
        }
    }

    pub(super) fn advance(&self, bytes: u64, total: Option<u64>) {
        if let (Some(total), None) = (total, self.bar.length()) {
            self.bar.set_length(total);
            self.bar.set_style(bar_style());
        }
        self.bar.inc(bytes);
        if !self.log_fallback {
            return;
        }
        let elapsed_ms = self.bar.elapsed().as_millis() as u64;
        let last = self.last_log_ms.load(Ordering::Relaxed);
        let due = elapsed_ms.saturating_sub(last) >= PROGRESS_LOG_INTERVAL_MS;
        // The CAS elects one logger per interval under concurrent chunks.
        if due
            && self
                .last_log_ms
                .compare_exchange(last, elapsed_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.log("downloading");
        }
    }

    pub(super) fn finish(&self, success: bool) {
        if self.log_fallback && success {
            self.log("downloaded");
        }
        self.bar.finish_and_clear();
        multi_progress().remove(&self.bar);
    }

    fn log(&self, state: &str) {
        let gb = |bytes: u64| bytes as f64 / 1e9;
        let done = self.bar.position();
        let speed_mb_s = self.bar.per_sec() / 1e6;
        let progress = match self.bar.length() {
            Some(total) => format!("{:.2}/{:.2} GB", gb(done), gb(total)),
            None => format!("{:.2} GB", gb(done)),
        };
        tracing::info!(
            file = %self.name,
            progress = %progress,
            speed = %format!("{speed_mb_s:.1} MB/s"),
            "{state}"
        );
    }
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg} {bar:30} {bytes}/{total_bytes} {bytes_per_sec} {eta}")
        .expect("invalid progress template")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} {msg} {bytes} {bytes_per_sec}")
        .expect("invalid progress template")
}
