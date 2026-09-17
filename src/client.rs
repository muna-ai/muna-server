/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::DashMap;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use muna::client::{
    Client, DownloadDurability, DownloadProgressFn,
    RequestInput, Result, SseStream
};
use muna::MunaClient;

/// Muna client for server use: wraps the default [`MunaClient`] and
/// overrides `download` with per-path single-flight, eventual durability,
/// and progress presentation (an indicatif bar on a TTY, throttled tracing
/// lines in containers/CI).
///
/// The registry already single-flights loads per tag, but two *different*
/// tags loading concurrently can share a resource file (e.g. `libtvm_ffi.so`);
/// the single-flight guard prevents
/// concurrent writes to one destination path. The lock map is
/// process-global (see [`download_locks`]) because the server constructs
/// one keyed client per model, and per-instance maps would not protect
/// across them.
///
/// Durability: fleet nodes bootstrap onto a fresh disk every time, so a
/// cache file torn by a crash is never read back. What matters is how soon
/// the engine can stage weights from page cache, and the default client's
/// fsync-before-rename makes that wait for the disk to absorb every byte.
/// On some network root volumes (~130 MB/s) that is the difference between
/// ~30 s and ~4 min for a 30 GB model, so downloads here are published
/// with [`DownloadDurability::Eventual`].
pub(crate) struct ServerClient {
    inner: MunaClient,
    /// Dedicated CDN client for `download`. The API client's pool is
    /// unrelated to CDN traffic, and the durability policy lives with the
    /// caller of `muna::download`, not with `MunaClient`.
    http: reqwest::Client,
}

impl ServerClient {

    /// Create a server client bound to a specific access key -- e.g. the
    /// per-model deployment key delivered with a residency directive.
    /// `None` falls back to `$MUNA_ACCESS_KEY` (standalone `muna deploy`
    /// behavior).
    pub fn with_key(key: Option<String>) -> Self {
        let access_key = key.or_else(|| std::env::var("MUNA_ACCESS_KEY").ok());
        let url = std::env::var("MUNA_API_URL").ok();
        Self {
            inner: MunaClient::new(access_key.as_deref(), url.as_deref()),
            http: cdn_client(),
        }
    }
}

/// HTTP client for resource downloads.
fn cdn_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("muna-server")
        .build()
        .expect("failed to build reqwest client")
}

/// Process-global in-flight download locks, keyed by destination path.
/// Shared across every `ServerClient` instance so concurrent loads through
/// different per-model clients still single-flight shared resource files.
fn download_locks() -> &'static DashMap<PathBuf, Arc<tokio::sync::Mutex<()>>> {
    static LOCKS: OnceLock<DashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>> = OnceLock::new();
    LOCKS.get_or_init(DashMap::new)
}

#[async_trait]
impl Client for ServerClient {

    fn url(&self) -> &str {
        self.inner.url()
    }

    fn cache_path(&self) -> &Path {
        self.inner.cache_path()
    }

    async fn request(&self, input: RequestInput) -> Result<serde_json::Value> {
        Client::request(&self.inner, input).await
    }

    async fn stream(&self, input: RequestInput) -> Result<SseStream<serde_json::Value>> {
        Client::stream(&self.inner, input).await
    }

    async fn fetch(&self, url: &str) -> Result<Vec<u8>> {
        self.inner.fetch(url).await
    }

    async fn download(
        &self,
        url: &str,
        path: &Path,
        progress: Option<DownloadProgressFn>,
    ) -> Result<()> {
        let lock = download_locks()
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let result = {
            let _guard = lock.lock().await;
            // A waiter that lost the race finds the file already in place.
            if tokio::fs::try_exists(path).await.unwrap_or(false) {
                Ok(())
            } else if let Some(callback) = progress {
                // A caller-provided callback wins over our presentation.
                self.download_eventual(url, path, callback).await
            } else {
                let presentation = DownloadProgress::new(path);
                let callback = {
                    let presentation = presentation.clone();
                    Arc::new(move |increment: u64, total: Option<u64>| {
                        presentation.advance(increment, total)
                    }) as DownloadProgressFn
                };
                let result = self.download_eventual(url, path, callback).await;
                presentation.finish(result.is_ok());
                result
            }
        };
        // Best-effort cleanup: drop the map entry once no download holds it.
        download_locks()
            .remove_if(&path.to_path_buf(), |_, lock| Arc::strong_count(lock) <= 2);
        result
    }

    async fn upload(&self, path: &Path) -> Result<String> {
        self.inner.upload(path).await
    }
}

impl ServerClient {

    /// Publish as soon as the bytes are in page cache; see the type-level
    /// durability note.
    async fn download_eventual(
        &self,
        url: &str,
        path: &Path,
        progress: DownloadProgressFn,
    ) -> Result<()> {
        muna::download(
            &self.http,
            url,
            path,
            Some(progress),
            DownloadDurability::Eventual
        ).await
    }
}

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
struct DownloadProgress {
    name: String,
    bar: ProgressBar,
    /// Bar-elapsed milliseconds of the last emitted log line.
    last_log_ms: Arc<AtomicU64>,
    /// Emit tracing lines instead of the (hidden) bar.
    log_fallback: bool,
}

impl DownloadProgress {

    fn new(path: &Path) -> Self {
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

    fn advance(&self, bytes: u64, total: Option<u64>) {
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

    fn finish(&self, success: bool) {
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Serve 1 KiB at `/resource`, counting every request (probe included)
    /// and delaying the response to widen the concurrent-download window.
    async fn start_resource_server(hits: Arc<AtomicUsize>) -> String {
        use axum::routing::get;
        let app = axum::Router::new().route(
            "/resource",
            get(move || {
                let hits = hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    vec![7u8; 1024]
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/resource", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        url
    }

    fn test_client() -> ServerClient {
        ServerClient {
            inner: MunaClient::new(None, None),
            http: cdn_client(),
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir()
            .join(format!("muna-server-client-test-{}", std::process::id()))
            .join(name)
    }

    #[tokio::test]
    async fn concurrent_downloads_of_one_path_single_flight() {
        let hits = Arc::new(AtomicUsize::new(0));
        let url = start_resource_server(hits.clone()).await;
        let client = test_client();

        // Baseline: how many requests one download costs (probe + body).
        let baseline_path = temp_path("single-flight-baseline.bin");
        let _ = std::fs::remove_file(&baseline_path);
        Client::download(&client, &url, &baseline_path, None).await.unwrap();
        let baseline = hits.swap(0, Ordering::SeqCst);
        assert!(baseline > 0);

        // Two concurrent downloads of one destination: the loser of the race
        // waits on the winner's lock, finds the file, and never transfers.
        let path = temp_path("single-flight-concurrent.bin");
        let _ = std::fs::remove_file(&path);
        let (a, b) = tokio::join!(
            Client::download(&client, &url, &path, None),
            Client::download(&client, &url, &path, None),
        );
        a.unwrap();
        b.unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), baseline);
        assert_eq!(std::fs::read(&path).unwrap(), vec![7u8; 1024]);
    }

    #[tokio::test]
    async fn eventual_download_publishes_complete_file_without_leftovers() {
        // The server's eventual-durability path must still deliver a complete
        // payload at the destination and leave no temp file behind.
        let hits = Arc::new(AtomicUsize::new(0));
        let url = start_resource_server(hits).await;
        let client = test_client();
        let dir = std::env::temp_dir().join(format!(
            "muna-server-eventual-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("resource.bin");
        Client::download(&client, &url, &path, None).await.unwrap();
        let payload = std::fs::read(&path).unwrap();
        let names: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(payload, vec![7u8; 1024]);
        assert_eq!(names, vec![std::ffi::OsString::from("resource.bin")]);
    }
}
