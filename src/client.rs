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

use crate::serving::preparation::{FailureStage, PreparationTracker};

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
    preparation: Option<(String, PreparationTracker)>,
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
            preparation: None,
        }
    }

    pub(crate) fn tracking(key: Option<String>, tag: &str, tracker: PreparationTracker) -> Self {
        tracker.begin(tag);
        let mut client = Self::with_key(key);
        client.preparation = Some((tag.to_owned(), tracker));
        client
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

    fn inference_url(&self) -> &str {
        self.inner.inference_url()
    }

    fn cache_path(&self) -> &Path {
        self.inner.cache_path()
    }

    async fn request(&self, input: RequestInput) -> Result<serde_json::Value> {
        let base = input.base_url.as_deref().unwrap_or(self.url());
        let resolving = input.path == "/predictions" && input.method == reqwest::Method::POST &&
            base.trim_end_matches('/') == self.url().trim_end_matches('/') &&
            self.preparation.as_ref().is_some_and(|(tag, _)|
                input.body.as_ref().and_then(|body| body.get("tag")).and_then(|tag| tag.as_str()) == Some(tag.as_str()));
        let response = Client::request(&self.inner, input).await;
        if resolving {
            if let Some((tag, tracker)) = &self.preparation {
                match &response {
                    Ok(body) => {
                        let resources = body.get("resources").cloned().unwrap_or(serde_json::json!([]));
                        if let Ok(resources) = serde_json::from_value::<Vec<muna::types::PredictionResource>>(resources) {
                            tracker.manifest(tag, self.cache_path(), &resources).await;
                        }
                    }
                    Err(_) => tracker.failed(tag, FailureStage::Resolve),
                }
            }
        }
        response
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
            // All keyed clients observe the same path, including waiters whose
            // shared resource was downloaded by a different predictor.
            if let Ok(metadata) = tokio::fs::metadata(path).await {
                if let Some((_, tracker)) = &self.preparation {
                    tracker.finished(path, Some(metadata.len()));
                }
                Ok(())
            } else {
                if let Some((_, tracker)) = &self.preparation { tracker.started(path); }
                let presentation = progress.is_none().then(|| DownloadProgress::new(path));
                let observer = self.preparation.clone();
                let observed_path = path.to_path_buf();
                let draw = presentation.clone();
                let callback = Arc::new(move |increment: u64, total: Option<u64>| {
                    if let Some((_, tracker)) = &observer { tracker.advance(&observed_path, increment, total); }
                    if let Some(callback) = &progress { callback(increment, total); }
                    if let Some(draw) = &draw { draw.advance(increment, total); }
                }) as DownloadProgressFn;
                let result = self.download_eventual(url, path, callback).await;
                if let Some(presentation) = presentation { presentation.finish(result.is_ok()); }
                if let Some((tag, tracker)) = &self.preparation {
                    let bytes = if result.is_ok() {
                        tokio::fs::metadata(path).await.ok().map(|m| m.len())
                    } else { None };
                    tracker.finished(path, bytes);
                    if result.is_err() { tracker.failed(tag, FailureStage::Download); }
                }
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
            preparation: None,
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

#[cfg(test)]
mod preparation_tests {
    use super::*;
    use crate::notifications::NotificationCenter;
    use crate::serving::preparation::PreparationStage;

    #[tokio::test]
    async fn manifests_are_observed_before_downloads_and_sidecar_resolutions_do_not_replace_them() {
        use axum::routing::post;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new().route("/predictions", post(|| async {
            axum::Json(serde_json::json!({ "resources": [
                { "type": "dso", "url": "https://cdn.example/resources/progress-test-unique?signature=secret", "size": 100 }
            ] }))
        }));
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let mut client = ServerClient::tracking(None, "@a/main", tracker.clone());
        client.inner = MunaClient::new(None, Some(&url));
        Client::request(&client, RequestInput::post("/predictions").body(serde_json::json!({"tag":"@a/sidecar"}))).await.unwrap();
        assert!(tracker.snapshot("@a/main").unwrap().resources_total.is_none());
        Client::request(&client, RequestInput::post("/predictions").body(serde_json::json!({"tag":"@a/main"}))).await.unwrap();
        let progress = tracker.snapshot("@a/main").unwrap();
        assert_eq!(progress.resources_total, Some(1));
        assert_eq!(progress.bytes_total, Some(100));
        assert_eq!(progress.stage, PreparationStage::Downloading);
        task.abort();
    }

    #[tokio::test]
    async fn supplied_callbacks_do_not_disable_tracking_and_waiters_observe_the_same_file() {
        use std::sync::atomic::AtomicUsize;
        let hits = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/resource", listener.local_addr().unwrap());
        let request_hits = hits.clone();
        let app = axum::Router::new().route("/resource", axum::routing::get(move || {
            let hits = request_hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
                vec![7u8; 1024]
            }
        }));
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let root = std::env::temp_dir().join(format!("muna-progress-download-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let path = root.join("resource");
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let resource = muna::types::PredictionResource { kind: "dso".into(), url: url.clone(), name: None, size: Some(1024) };
        tracker.manifest("a", &root, &[resource.clone()]).await;
        tracker.manifest("b", &root, &[resource]).await;
        let a = ServerClient::tracking(None, "a", tracker.clone());
        let b = ServerClient::tracking(None, "b", tracker.clone());
        let callback_bytes = Arc::new(AtomicU64::new(0));
        let observed = callback_bytes.clone();
        let callback: DownloadProgressFn = Arc::new(move |bytes, _| { observed.fetch_add(bytes, Ordering::Relaxed); });
        let (first, second) = tokio::join!(Client::download(&a, &url, &path, Some(callback)), Client::download(&b, &url, &path, None));
        first.unwrap(); second.unwrap();
        assert_eq!(callback_bytes.load(Ordering::Relaxed), 1024);
        for tag in ["a", "b"] {
            let progress = tracker.snapshot(tag).unwrap();
            assert_eq!(progress.resources_completed, 1);
            assert_eq!(progress.bytes_available, 1024);
            assert!(progress.active_resources.is_empty());
        }
        let downloaded_hits = hits.load(Ordering::SeqCst);
        Client::download(&b, &url, &path, None).await.unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), downloaded_hits);
        task.abort();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn resolution_and_download_failures_are_structured_and_sanitized() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new().fallback(|| async { axum::http::StatusCode::FORBIDDEN });
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let mut client = ServerClient::tracking(None, "a", tracker.clone());
        client.inner = MunaClient::new(None, Some(&base));
        assert!(Client::request(&client, RequestInput::post("/predictions").body(serde_json::json!({"tag":"a"}))).await.is_err());
        assert_eq!(tracker.failure("a").unwrap().stage, FailureStage::Resolve);
        let path = std::env::temp_dir().join(format!("muna-progress-failed-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert!(Client::download(&client, &format!("{base}/resource?signature=secret"), &path, None).await.is_err());
        let failure = tracker.failure("a").unwrap();
        assert_eq!(failure.stage, FailureStage::Download);
        assert!(failure.retrying);
        assert!(!serde_json::to_string(&failure).unwrap().contains("secret"));
        task.abort();
    }
}
