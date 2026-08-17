/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Shared harness for the LLM-serving integration tests.
//!
//! - `StubControlPlane`: in-process axum router that impersonates the
//!   control plane. Tests mutate its directives directly; the spawned
//!   muna-server reconciles against them on every heartbeat.
//! - `ServerGuard`: spawns the real `muna-server` binary against the stub
//!   and kills it on drop, so panicking tests never leak processes.
//! - `wait_for`: polling helper used instead of bare sleeps.
//! - Remote-value helpers for the `/v1/predictions/remote` wire format
//!   (inline base64 `data:` URLs).

#![allow(dead_code)]

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Path, State};
use axum::routing::post;
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Stub control plane
// ---------------------------------------------------------------------------

/// Directives returned to the node on every heartbeat. Tests mutate these
/// through `StubControlPlane::state()`.
#[derive(Default)]
pub struct Directives {
    pub load_models: Vec<String>,
    pub prefetch_models: Vec<String>,
    pub unload_models: Vec<String>,
    pub event_callback_urls: Vec<String>,
    pub drain: bool,
}

#[derive(Default)]
pub struct StubState {
    /// Every `NodeStatus` payload the node has POSTed, in arrival order.
    pub heartbeats: Vec<Value>,
    /// Directives echoed back on the next heartbeat.
    pub directives: Directives,
    /// Every `RelayBatch` POSTed to the `/kv` edge callback.
    pub kv_batches: Vec<Value>,
    /// One-shot: reply `need_snapshot: true` to the next `/kv` POST.
    pub need_snapshot_armed: bool,
}

pub struct StubControlPlane {
    state: Arc<Mutex<StubState>>,
    port: u16,
}

impl StubControlPlane {

    /// Start the stub on an ephemeral port.
    pub async fn start() -> Self {
        let state = Arc::new(Mutex::new(StubState::default()));
        let router = Router::new()
            .route("/v1/nodes/{node_id}/heartbeat", post(heartbeat))
            .route("/kv", post(kv))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind stub control plane");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, router).await.expect("stub control plane crashed");
        });
        Self { state, port }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// The `/kv` edge-callback URL, for the `event_callback_urls` directive.
    pub fn kv_callback_url(&self) -> String {
        format!("http://127.0.0.1:{}/kv", self.port)
    }

    /// Lock the mutable stub state (directives, recorded traffic).
    pub fn state(&self) -> std::sync::MutexGuard<'_, StubState> {
        self.state.lock().unwrap()
    }

    pub fn heartbeat_count(&self) -> usize {
        self.state().heartbeats.len()
    }

    pub fn last_heartbeat(&self) -> Option<Value> {
        self.state().heartbeats.last().cloned()
    }
}

async fn heartbeat(
    State(state): State<Arc<Mutex<StubState>>>,
    Path(_node_id): Path<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut state = state.lock().unwrap();
    state.heartbeats.push(payload);
    let directives = &state.directives;
    Json(json!({
        "load_models": directives.load_models,
        "prefetch_models": directives.prefetch_models,
        "unload_models": directives.unload_models,
        "event_callback_urls": directives.event_callback_urls,
        "drain": directives.drain,
    }))
}

async fn kv(
    State(state): State<Arc<Mutex<StubState>>>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    let mut state = state.lock().unwrap();
    state.kv_batches.push(payload);
    let need_snapshot = state.need_snapshot_armed;
    state.need_snapshot_armed = false;
    Json(json!({ "need_snapshot": need_snapshot }))
}

// ---------------------------------------------------------------------------
// Server spawn guard
// ---------------------------------------------------------------------------

pub struct ServerGuard {
    child: Child,
    port: u16,
}

impl ServerGuard {

    /// Spawn the real muna-server binary wired to the stub control plane,
    /// with 1s heartbeat + KV flush cadence, and wait until `/health` is up.
    pub async fn spawn(stub: &StubControlPlane) -> Self {
        Self::spawn_with_node_id(stub, "test-node").await
    }

    pub async fn spawn_with_node_id(
        stub: &StubControlPlane,
        node_id: &str
    ) -> Self {
        // Fresh data dir per guard so tests never touch (or observe) the
        // developer's real ~/.muna/server manifest.
        Self::spawn_with_data_dir(stub, node_id, &fresh_data_dir()).await
    }

    /// Spawn with an explicit `--data-dir`: restart tests reuse one dir
    /// across guards to prove the manifest survives the process.
    pub async fn spawn_with_data_dir(
        stub: &StubControlPlane,
        node_id: &str,
        data_dir: &std::path::Path
    ) -> Self {
        let port = ephemeral_port();
        let mut command = Command::new(env!("CARGO_BIN_EXE_muna-server"));
        command
            .arg("--port").arg(port.to_string())
            .arg("--control-plane-url").arg(stub.url())
            .arg("--node-id").arg(node_id)
            .arg("--heartbeat-interval").arg("1")
            .arg("--kv-flush-interval").arg("1")
            .arg("--data-dir").arg(data_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // muna-rs only reads $MUNA_ACCESS_KEY; splice in the key from the
        // crate-local .env so it is the only local setup step.
        if std::env::var("MUNA_ACCESS_KEY").is_err() {
            if let Some(key) = stored_access_key() {
                command.env("MUNA_ACCESS_KEY", key);
            }
        }
        let child = command.spawn().expect("failed to spawn muna-server");
        let guard = Self { child, port };
        let health = format!("{}/health", guard.url());
        let client = reqwest::Client::new();
        wait_for(Duration::from_secs(15), || {
            let client = client.clone();
            let health = health.clone();
            async move {
                client.get(&health).send().await.is_ok_and(|r| r.status().is_success())
            }
        })
        .await
        .expect("muna-server did not become healthy in time");
        guard
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Whether an access key is configured (environment or crate-local `.env`).
pub fn access_key_available() -> bool {
    resolved_access_key().is_some()
}

/// The access key the spawned server will use: environment first, then the
/// crate-local `.env`.
pub fn resolved_access_key() -> Option<String> {
    std::env::var("MUNA_ACCESS_KEY")
        .ok()
        .filter(|key| !key.is_empty())
        .or_else(stored_access_key)
}

/// Whether a fake predictor has been pushed, via `muna.predictors.retrieve`.
/// `None` means the availability could not be determined (API/network
/// failure); callers should proceed and let the test surface the real error
/// rather than skip silently.
pub async fn predictor_available(tag: &str) -> Option<bool> {
    let key = resolved_access_key()?;
    let muna = muna::Muna::new(Some(&key), None);
    match muna.predictors.retrieve(tag).await {
        Ok(predictor) => Some(predictor.is_some()),
        Err(_) => None,
    }
}

/// Access key from the crate-local `.env` file (see `.env.example`).
pub fn stored_access_key() -> Option<String> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".env");
    let contents = std::fs::read_to_string(path).ok()?;
    contents.lines().find_map(|line| {
        let line = line.trim();
        let value = line.strip_prefix("MUNA_ACCESS_KEY=")?;
        let value = value.trim().trim_matches('"').trim_matches('\'');
        (!value.is_empty()).then(|| value.to_string())
    })
}

/// A unique server data dir under the OS temp dir (holds `predictors.json`).
pub fn fresh_data_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static UNIQUE: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "muna-server-test-data-{}-{}",
        std::process::id(),
        UNIQUE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn ephemeral_port() -> u16 {
    // Bind-then-drop: the port stays free long enough for the child to
    // claim it (standard test-harness race, acceptable locally).
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to reserve port");
    listener.local_addr().unwrap().port()
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

/// Poll `predicate` every 100ms until it returns true or `timeout` elapses.
pub async fn wait_for<F, Fut>(timeout: Duration, mut predicate: F) -> Result<(), String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if predicate().await {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!("condition not met within {timeout:?}"));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ---------------------------------------------------------------------------
// /v1/predictions/remote wire format helpers
// ---------------------------------------------------------------------------

fn encode_data_url(buffer: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", BASE64.encode(buffer))
}

fn decode_data_url(url: &str) -> Vec<u8> {
    let (_, encoded) = url
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,"))
        .expect("expected inline base64 data URL");
    BASE64.decode(encoded).expect("invalid base64 in data URL")
}

/// A `string`-typed remote value.
pub fn remote_string(value: &str) -> Value {
    json!({
        "data": encode_data_url(value.as_bytes(), "text/plain"),
        "dtype": "string",
    })
}

/// A `list`-typed remote value.
pub fn remote_list(items: Value) -> Value {
    let json = serde_json::to_string(&items).unwrap();
    json!({
        "data": encode_data_url(json.as_bytes(), "application/json"),
        "dtype": "list",
    })
}

/// Decode a `dict`- or `list`-typed remote value from prediction results.
pub fn decode_remote_json(remote: &Value) -> Value {
    let dtype = remote["dtype"].as_str().expect("remote value has no dtype");
    assert!(
        dtype == "dict" || dtype == "list",
        "expected dict/list remote value, got {dtype}"
    );
    let bytes = decode_data_url(remote["data"].as_str().expect("remote value has no data"));
    serde_json::from_slice(&bytes).expect("remote value is not valid JSON")
}

/// Decode a `string`-typed remote value.
pub fn decode_remote_string(remote: &Value) -> String {
    assert_eq!(remote["dtype"].as_str(), Some("string"));
    let bytes = decode_data_url(remote["data"].as_str().expect("remote value has no data"));
    String::from_utf8(bytes).expect("remote string is not UTF-8")
}
