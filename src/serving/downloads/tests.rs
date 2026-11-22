/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Local HTTP tests exercise actual range I/O, scheduling and publication.

use super::*;
use std::sync::atomic::{AtomicU64, AtomicU8};

use axum::body::Body;
use axum::extract::{Path as RoutePath, State as RouteState};
use axum::http::{HeaderMap, Response, StatusCode};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

mod scheduling;
mod transfers;

#[derive(Clone)]
struct Resource {
    bytes: Vec<u8>,
    // 0: ranges; 1: stream; 2: unknown-length stream; >=3: invalid response.
    mode: Arc<AtomicU8>,
    gate: Option<Arc<Semaphore>>,
}

impl Resource {
    fn new(size: usize) -> Self {
        Self {
            bytes: (0..size).map(|i| (i % 251) as u8).collect(),
            mode: Arc::default(),
            gate: None,
        }
    }

    fn gated(size: usize) -> Self {
        Self {
            gate: Some(Arc::new(Semaphore::new(0))),
            ..Self::new(size)
        }
    }
}

#[derive(Clone, Debug)]
struct Request {
    name: String,
    start: u64,
    end: u64,
    probe: bool,
}

struct Origin {
    files: HashMap<String, Resource>,
    requests: Mutex<Vec<Request>>,
}

struct Fixture {
    root: PathBuf,
    url: String,
    origin: Arc<Origin>,
    http: reqwest::Client,
    server: JoinHandle<()>,
}

impl Fixture {
    async fn new(files: &[(&str, Resource)]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "muna-ranges-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let origin = Arc::new(Origin {
            files: files
                .iter()
                .map(|(name, resource)| (name.to_string(), resource.clone()))
                .collect(),
            requests: Mutex::default(),
        });
        let app = axum::Router::new()
            .route("/{name}", axum::routing::get(serve))
            .with_state(origin.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            root,
            url,
            origin,
            http: reqwest::Client::new(),
            server,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
    fn url(&self, name: &str) -> String {
        format!("{}/{name}", self.url)
    }

    fn bodies(&self, name: &str) -> Vec<Request> {
        self.origin
            .requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.name == name && !request.probe)
            .cloned()
            .collect()
    }

    fn enqueue(
        &self,
        downloads: &Downloads,
        name: &str,
        priority: Priority,
        progress: Option<DownloadProgressFn>,
        tracker: Option<PreparationTracker>,
    ) -> JoinHandle<Result<(), String>> {
        let downloads = downloads.clone();
        let http = self.http.clone();
        let path = self.path(name);
        let url = self.url(name);
        tokio::spawn(async move {
            downloads
                .download(&http, &url, &path, priority, progress, tracker)
                .await
        })
    }

    async fn model(&self, tag: &str, names: &[&str]) -> (Priority, PreparationTracker) {
        let tracker =
            PreparationTracker::new(Arc::new(crate::notifications::NotificationCenter::default()))
                .scoped(tag);
        let resources = names
            .iter()
            .map(|name| {
                serde_json::from_value(serde_json::json!({
                    "type":"bin", "url":self.url(name), "size":self.origin.files[*name].bytes.len(),
                }))
                .unwrap()
            })
            .collect::<Vec<_>>();
        tracker.manifest(tag, &self.root, &resources).await;
        let priority = Priority::default();
        priority.track(tag, tracker.clone());
        (priority, tracker)
    }

    fn enqueue_model(
        &self,
        downloads: &Downloads,
        names: &[&str],
        priority: Priority,
        tracker: PreparationTracker,
    ) -> JoinHandle<Result<(), String>> {
        let downloads = downloads.clone();
        let http = self.http.clone();
        let resources: Vec<_> = names
            .iter()
            .map(|name| (self.url(name), self.path(name)))
            .collect();
        tokio::spawn(async move {
            for (url, path) in resources {
                downloads
                    .download(
                        &http,
                        &url,
                        &path,
                        priority.clone(),
                        None,
                        Some(tracker.clone()),
                    )
                    .await?;
            }
            Ok(())
        })
    }

    fn assert_clean(&self) {
        assert!(std::fs::read_dir(&self.root).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".part")));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

async fn serve(
    RoutePath(name): RoutePath<String>,
    RouteState(origin): RouteState<Arc<Origin>>,
    headers: HeaderMap,
) -> Response<Body> {
    let Some(resource) = origin.files.get(&name) else {
        return Response::builder().status(404).body(Body::empty()).unwrap();
    };
    let mode = resource.mode.load(Ordering::Relaxed);
    let total = resource.bytes.len();
    let range = headers
        .get("range")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("bytes="))
        .and_then(|value| value.split_once('-'))
        .map(|(start, end)| {
            (
                start.parse::<usize>().unwrap(),
                end.parse::<usize>().unwrap(),
            )
        });
    let (start, end) = range.unwrap_or((0, total.saturating_sub(1)));
    let probe = range == Some((0, 0));
    origin.requests.lock().unwrap().push(Request {
        name,
        start: start as u64,
        end: end as u64,
        probe,
    });
    if !probe {
        if let Some(gate) = &resource.gate {
            gate.acquire().await.unwrap().forget();
        }
    }
    if mode == 9 && !probe {
        return Response::builder().status(503).body(Body::empty()).unwrap();
    }
    if total == 0 {
        return Response::builder()
            .status(416)
            .header("content-range", "bytes */0")
            .body(Body::empty())
            .unwrap();
    }
    if mode == 1 || mode == 2 || (mode == 4 && !probe) {
        let builder = Response::builder().status(200);
        return if mode == 2 {
            let bytes = resource.bytes.clone();
            builder
                .body(Body::from_stream(futures_util::stream::once(async move {
                    Ok::<_, std::convert::Infallible>(bytes)
                })))
                .unwrap()
        } else {
            builder.body(Body::from(resource.bytes.clone())).unwrap()
        };
    }
    let end = end.min(total - 1);
    let mut bytes = resource.bytes[start..=end].to_vec();
    if mode == 5 && !probe {
        bytes.pop();
    }
    if mode == 6 && !probe {
        bytes.push(0);
    }
    let range_start = if mode == 3 && !probe {
        start + 1
    } else {
        start
    };
    let range_total = if mode == 7 && !probe {
        total + 1
    } else {
        total
    };
    let mut builder = Response::builder()
        .status(StatusCode::PARTIAL_CONTENT)
        .header("etag", "\"fixture-v1\"");
    if mode != 8 || probe {
        builder = builder.header(
            "content-range",
            format!("bytes {range_start}-{end}/{range_total}"),
        );
    }
    builder.body(Body::from(bytes)).unwrap()
}

fn downloads() -> Downloads {
    Downloads(Arc::new(Inner {
        limits: Limits {
            range: 4,
            quantum: 8,
        },
        ..Inner::default()
    }))
}

async fn until(condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("condition did not become true");
}

async fn registered(downloads: &Downloads, path: &Path, subscribers: usize) {
    until(|| {
        downloads
            .0
            .queue
            .lock()
            .unwrap()
            .jobs
            .get(path)
            .is_some_and(|job| job.state.lock().unwrap().subscribers.len() == subscribers)
    })
    .await;
}

async fn complete(task: JoinHandle<Result<(), String>>) {
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}
