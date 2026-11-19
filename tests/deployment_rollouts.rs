/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Full heartbeat -> disk preparation -> rollout acknowledgement, using only
//! localhost services. No published predictors, credentials, or GPUs required.

mod common;

use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::HeaderMap, routing::{get, post}, Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use common::{wait_for, ServerGuard, StubControlPlane};

const TAG: &str = "@test/deployment";

struct Api {
    url: String,
    resolutions: AtomicUsize,
    signatures: AtomicUsize,
    downloads: AtomicUsize,
}

async fn resolve(State(api): State<Arc<Api>>, headers: HeaderMap) -> Json<Value> {
    api.resolutions.fetch_add(1, Ordering::SeqCst);
    assert!(matches!(headers["authorization"].to_str().unwrap(), "Bearer first-key" | "Bearer second-key"));
    Json(json!({
        "id": "pred_local",
        "tag": TAG,
        "created": "2026-11-19T00:00:00Z",
        "configuration": null,
        "resources": [
            {
                "type": "bin",
                "url": format!("{}/resources/weights", api.url),
                "size": 4
            }
        ]
    }))
}

async fn predictor(State(api): State<Arc<Api>>) -> Json<Value> {
    api.signatures.fetch_add(1, Ordering::SeqCst);
    Json(json!({
        "tag": TAG,
        "owner": { "username": "test" },
        "name": "deployment",
        "status": "active",
        "access": "private",
        "signature": {
            "inputs": [],
            "outputs": []
        },
        "created": "2026-11-19T00:00:00Z"
    }))
}

async fn resource(State(api): State<Arc<Api>>) -> &'static str {
    api.downloads.fetch_add(1, Ordering::SeqCst);
    "data"
}

fn disk(stub: &StubControlPlane, key: &str) {
    stub.state().directives.models.insert(TAG.into(), json!({ "tag": TAG, "residency": "disk", "key": key }));
}

async fn cached(stub: &StubControlPlane, key: &str) {
    let hash = format!("{:x}", Sha256::digest(key.as_bytes()));
    wait_for(Duration::from_secs(10), || async {
        stub.last_heartbeat().is_some_and(|body| body["models"].as_array().unwrap().iter().any(|model|
            model["tag"] == TAG && model["state"] == "cached" && model["key_hash"] == hash && model["disk_ready"] == true))
    }).await.expect("current deployment never acknowledged disk readiness");
    let wire = stub.last_heartbeat().unwrap().to_string();
    assert!(!wire.contains("first-key") && !wire.contains("second-key"));
}

#[tokio::test(flavor = "multi_thread")]
async fn key_rotation_refreshes_metadata_preserves_files_and_acknowledges_the_right_attempt() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api = Arc::new(Api {
        url: format!("http://{}", listener.local_addr().unwrap()),
        resolutions: AtomicUsize::new(0), signatures: AtomicUsize::new(0), downloads: AtomicUsize::new(0),
    });
    let router = Router::new().route("/predictions", post(resolve)).route("/predictors/@test/deployment", get(predictor))
        .route("/resources/weights", get(resource)).with_state(api.clone());
    let http = tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    let home = std::env::temp_dir().join(format!("muna-rollout-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
    let stub = StubControlPlane::start().await;
    disk(&stub, "first-key");
    let server = ServerGuard::spawn_with_api(&stub, &api.url, &home).await;
    cached(&stub, "first-key").await;
    let downloads = api.downloads.load(Ordering::SeqCst);
    assert!(downloads > 0);
    assert_eq!(api.resolutions.load(Ordering::SeqCst), 1);
    // Reasserting the goal is an idempotent no-op, rather than repeated API work.
    let beats = stub.heartbeat_count();
    wait_for(Duration::from_secs(5), || async { stub.heartbeat_count() >= beats + 2 }).await.unwrap();
    assert_eq!(api.resolutions.load(Ordering::SeqCst), 1);

    disk(&stub, "second-key");
    cached(&stub, "second-key").await;
    assert_eq!(api.resolutions.load(Ordering::SeqCst), 2);
    assert_eq!(api.signatures.load(Ordering::SeqCst), 2);
    assert_eq!(api.downloads.load(Ordering::SeqCst), downloads);

    stub.set_residency(TAG, "none");
    wait_for(Duration::from_secs(10), || async {
        stub.last_heartbeat().is_some_and(|body| body["models"].as_array().unwrap().is_empty())
    }).await.unwrap();
    assert_eq!(std::fs::read(home.join("cache/weights")).unwrap(), b"data");
    // A fresh preparation (new SDK instances) reuses the persisted response.
    disk(&stub, "first-key");
    cached(&stub, "first-key").await;
    assert_eq!(api.resolutions.load(Ordering::SeqCst), 2);
    assert_eq!(api.signatures.load(Ordering::SeqCst), 2);
    assert_eq!(api.downloads.load(Ordering::SeqCst), downloads);
    drop(server);
    http.abort();
    std::fs::remove_dir_all(home).unwrap();
}
