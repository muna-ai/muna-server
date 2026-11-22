/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use muna::client::{
    Client, DownloadProgressFn,
    RequestInput, Result, SseStream
};
use muna::MunaClient;

use crate::serving::preparation::{FailureStage, PreparationTracker};
use crate::serving::downloads::{Downloads, Priority};

mod metadata;
mod preparation;
pub(crate) use metadata::key_hash;
use metadata::DeploymentCache;

/// Server transport. All keyed clients on a node share a resource scheduler;
/// only the active transfer updates physical bytes in the preparation tracker.
/// Cache publication uses eventual durability so loading can read page cache
/// without waiting for a slow ephemeral disk's writeback.
pub(crate) struct ServerClient {
    inner: MunaClient,
    deployment: Option<DeploymentCache>,
    /// Dedicated CDN client for `download`. The API client's pool is
    /// unrelated to CDN traffic. The server schedules bounded range batches
    /// and publishes ephemeral cache files without per-resource fsync.
    http: reqwest::Client,
    preparation: Option<(String, PreparationTracker)>,
    downloads: Downloads,
    priority: Priority,
    reservations: Mutex<Vec<crate::serving::downloads::disk::Reservation>>,
    available_gpus: Option<u32>,
}

impl ServerClient {

    /// Create a server client bound to a specific access key -- e.g. the
    /// per-model deployment key delivered with a residency directive.
    /// `None` falls back to `$MUNA_ACCESS_KEY` (standalone `muna deploy`
    /// behavior).
    pub fn with_key(key: Option<String>) -> Self {
        let access_key = key.clone().or_else(|| std::env::var("MUNA_ACCESS_KEY").ok());
        let url = std::env::var("MUNA_API_URL").ok();
        let inner = MunaClient::new(access_key.as_deref(), url.as_deref());
        let deployment = key.map(|key| DeploymentCache::new(inner.cache_path(), &key));
        Self {
            inner,
            deployment,
            http: cdn_client(),
            preparation: None,
            downloads: Downloads::default(),
            priority: Priority::default(),
            reservations: Mutex::default(),
            available_gpus: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn tracking(key: Option<String>, tag: &str, tracker: PreparationTracker) -> Self {
        Self::scheduled(key, tag, tracker, Downloads::default(), Priority::default())
    }

    pub(crate) fn scheduled(
        key: Option<String>,
        tag: &str,
        tracker: PreparationTracker,
        downloads: Downloads,
        priority: Priority
    ) -> Self {
        let tracker = tracker.scoped(tag);
        priority.track(tag, tracker.clone());
        let mut client = Self::with_key(key);
        client.preparation = Some((tag.to_owned(), tracker));
        client.downloads = downloads;
        client.priority = priority;
        client
    }

    pub(crate) fn with_gpu_capacity(mut self, count: Option<u32>) -> Self {
        self.available_gpus = count;
        self
    }
}

/// HTTP client for resource downloads.
fn cdn_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("muna-server")
        .build()
        .expect("failed to build reqwest client")
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
        self.deployment.as_ref().map(DeploymentCache::sdk).unwrap_or_else(|| self.inner.cache_path())
    }

    async fn request(&self, input: RequestInput) -> Result<serde_json::Value> {
        let base = input.base_url.as_deref().unwrap_or(self.url());
        let resolving = input.path == "/predictions" && input.method == reqwest::Method::POST &&
            base.trim_end_matches('/') == self.url().trim_end_matches('/') &&
            self.preparation.as_ref().is_some_and(|(tag, _)|
                input.body.as_ref().and_then(|body| body.get("tag")).and_then(|tag| tag.as_str()) == Some(tag.as_str()));
        let entry = self.deployment.as_ref().and_then(|cache| cache.entry(&input, self.url()));
        let cached = match (&self.deployment, &entry) {
            (Some(cache), Some(path)) => cache.read(path, &input).await,
            _ => None,
        };
        let fetched = cached.is_none();
        let response = match cached {
            Some(body) => Ok(body),
            None => Client::request(&self.inner, input).await,
        };
        if fetched {
            if let (Some(cache), Some(path), Ok(body)) = (&self.deployment, entry, &response) {
                cache.write(&path, body).await;
            }
        }
        if resolving {
            if let Some((tag, tracker)) = &self.preparation {
                match &response {
                    Ok(body) => self.admit(tag, tracker, body).await?,
                    Err(error) => tracker.resolution_failed(tag, error),
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
        let shared = self.deployment.as_ref().map(|cache| cache.shared_path(path)).unwrap_or_else(|| path.into());
        let result = self.downloads.download(
            &self.http, url, &shared, self.priority.clone(), progress,
            self.preparation.as_ref().map(|(_, tracker)| tracker.clone()),
        ).await;
        if let Some((tag, tracker)) = &self.preparation {
            if result.is_ok() {
                tracker.finished(&shared, tokio::fs::metadata(&shared).await.ok().map(|m| m.len()));
            } else {
                tracker.failed(tag, FailureStage::Download);
            }
        }
        result.map_err(muna::MunaError::Native)?;
        if path != shared {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| muna::MunaError::Native(error.to_string()))?;
            }
            match tokio::fs::hard_link(&shared, path).await {
                Ok(()) => {},
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
                Err(error) => return Err(muna::MunaError::Native(format!("Failed to link predictor resource: {error}"))),
            }
        }
        Ok(())
    }

    async fn upload(&self, path: &Path) -> Result<String> {
        self.inner.upload(path).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
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
            deployment: None,
            http: cdn_client(),
            preparation: None,
            downloads: Downloads::default(),
            priority: Priority::default(),
            reservations: Mutex::default(),
            available_gpus: None,
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
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
        let downloads = Downloads::default();
        let a = ServerClient::scheduled(None, "a", tracker.clone(), downloads.clone(), Priority::default());
        let b = ServerClient::scheduled(None, "b", tracker.clone(), downloads, Priority::default());
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
