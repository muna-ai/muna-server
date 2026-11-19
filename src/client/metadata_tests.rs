/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};

use axum::{extract::State, http::HeaderMap, routing::{get, post}, Json, Router};
use muna::client::Client;

use super::*;
use crate::client::{cdn_client, ServerClient};
use crate::notifications::NotificationCenter;
use crate::serving::preparation::PreparationTracker;

fn root() -> PathBuf {
    std::env::temp_dir().join(format!("muna-deployment-metadata-{}-{}", std::process::id(), WRITES.fetch_add(1, Ordering::Relaxed)))
}

fn request() -> RequestInput {
    RequestInput::post("/predictions").body(json!({ "tag": "@test/model", "target": "x86_64-unknown-linux-gnu", "computeArchitecture": null }))
}

fn prediction(id: &str) -> Value {
    json!({ "id": id, "tag": "@test/model", "created": "2026-11-19T00:00:00Z", "configuration": null, "resources": [] })
}

#[test]
fn identity_covers_all_selectors_and_ignores_json_field_order() {
    let cache = DeploymentCache::new(Path::new("/cache"), "key");
    let first = cache.entry(&request(), "http://api/v1").unwrap();
    let reordered = RequestInput::post("/predictions").body(json!({ "computeArchitecture": null, "target": "x86_64-unknown-linux-gnu", "tag": "@test/model" }));
    assert_eq!(Some(first.clone()), cache.entry(&reordered, "http://api/v1/"));
    for (field, value) in [("target", json!("aarch64-apple-darwin")), ("computeArchitecture", json!("sm_90")), ("configurationId", json!("other-runtime")), ("variant", json!("decode"))] {
        let mut input = request();
        input.body.as_mut().unwrap()[field] = value;
        assert_ne!(Some(first.clone()), cache.entry(&input, "http://api/v1"));
    }
    assert!(cache.entry(&request().base_url("http://inference/v1"), "http://api/v1").is_none());
    let mut partial = request();
    partial.body.as_mut().unwrap()["predictionId"] = json!("pinned");
    assert!(cache.entry(&partial, "http://api/v1").is_none());
    assert_ne!(cache.sdk(), DeploymentCache::new(Path::new("/cache"), "new-key").sdk());
    assert_eq!(key_hash("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[tokio::test]
async fn incomplete_resources_expired_tokens_and_corrupt_metadata_are_refetched() {
    let root = root();
    let cache = DeploymentCache::new(&root, "key");
    let path = cache.entry(&request(), "http://api").unwrap();
    let mut body = prediction("cached");
    body["resources"] = json!([{ "type": "bin", "url": "https://cdn/weights?expires=1", "size": 4 }]);
    cache.write(&path, &body).await;
    assert!(cache.read(&path, &request()).await.is_none());
    tokio::fs::write(root.join("weights"), b"data").await.unwrap();
    assert_eq!(cache.read(&path, &request()).await.unwrap()["id"], "cached");
    body["configuration"] = json!(format!("header.{}.signature", URL_SAFE_NO_PAD.encode(br#"{"exp":1}"#)));
    cache.write(&path, &body).await;
    assert!(cache.read(&path, &request()).await.is_none());
    tokio::fs::write(&path, b"incomplete JSON").await.unwrap();
    assert!(cache.read(&path, &request()).await.is_none());
    tokio::fs::remove_dir_all(root).await.unwrap();
}

struct Api {
    predictions: AtomicUsize,
    signatures: AtomicUsize,
}

async fn resolve(State(api): State<Arc<Api>>, headers: HeaderMap) -> Json<Value> {
    api.predictions.fetch_add(1, Ordering::SeqCst);
    Json(prediction(headers["authorization"].to_str().unwrap()))
}

async fn retrieve(State(api): State<Arc<Api>>) -> Json<Value> {
    api.signatures.fetch_add(1, Ordering::SeqCst);
    Json(json!({ "tag": "@test/model", "owner": { "username": "test" }, "name": "model", "status": "active", "access": "private", "signature": { "inputs": [], "outputs": [] }, "created": "2026-11-19T00:00:00Z" }))
}

fn client(root: &Path, key: &str, url: &str) -> ServerClient {
    let mut client = ServerClient::with_key(Some(key.into()));
    client.inner = muna::MunaClient::new(Some(key), Some(url));
    client.http = cdn_client();
    client.deployment = Some(DeploymentCache::new(root, key));
    client
}

#[tokio::test]
async fn a_new_client_reuses_persisted_metadata_but_a_new_deployment_resolves_again() {
    let root = root();
    let api = Arc::new(Api { predictions: AtomicUsize::new(0), signatures: AtomicUsize::new(0) });
    let router = Router::new().route("/predictions", post(resolve)).route("/predictors/@test/model", get(retrieve)).with_state(api.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap(); });
    for key in ["old-key", "old-key", "new-key", "new-key"] {
        let client = client(&root, key, &url);
        let muna = muna::Muna::with_client(Arc::new(client));
        let response = muna.predictions.create("@test/model", None, None, None, None, None, None, None).await.unwrap();
        assert_eq!(response.id, format!("Bearer {key}"));
        assert!(muna.predictors.retrieve("@test/model").await.unwrap().is_some());
    }
    assert_eq!(api.predictions.load(Ordering::SeqCst), 2);
    assert_eq!(api.signatures.load(Ordering::SeqCst), 2);
    // Different SDK metadata roots still share a physical resource transfer.
    let shared = root.join("weights");
    tokio::fs::write(&shared, b"weights").await.unwrap();
    for key in ["old-key", "new-key"] {
        let client = client(&root, key, &url);
        let destination = client.cache_path().join("weights");
        client.download("http://127.0.0.1:1/unreachable", &destination, None).await.unwrap();
        assert_eq!(tokio::fs::read(destination).await.unwrap(), b"weights");
    }
    server.abort();
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn cached_manifests_still_initialize_preparation_progress() {
    let root = root();
    let mut client = client(&root, "key", "http://127.0.0.1:1/v1");
    let cache = client.deployment.as_ref().unwrap();
    let path = cache.entry(&request(), client.url()).unwrap();
    cache.write(&path, &prediction("cached")).await;
    let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
    client.preparation = Some(("@test/model".into(), tracker.scoped("@test/model")));
    client.request(request()).await.unwrap();
    assert_eq!(tracker.snapshot("@test/model").unwrap().resources_total, Some(0));
    tokio::fs::remove_dir_all(root).await.unwrap();
}
