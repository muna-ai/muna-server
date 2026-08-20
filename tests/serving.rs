/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! End-to-end LLM-serving tests (local tier).
//!
//! Fake CPU predictors from `tests/predictors/` impersonate
//! OpenAI-compatible signatures, exercising the full path
//! HTTP handler -> registry -> dispatcher -> muna FFI -> compiled binary
//! against an in-process stub control plane.
//!
//! Real-engine behavior lives in `tests/llm_gate.rs`.
//!
//! Tests that need a fake predictor self-skip (with a loud message) when
//! the tag has not been pushed yet; see `tests/predictors/README.md` for
//! the one-time compile + push commands.

mod common;

use std::time::Duration;

use serde_json::{json, Value};

use common::{
    decode_remote_json, remote_list, remote_string,
    wait_for, ServerGuard, StubControlPlane,
};

// Fake predictor tags (pushed from tests/predictors/, see its README).
const TAG_CHAT: &str = "@muna/test-openai-chat";
const TAG_EMBEDDINGS: &str = "@muna/test-openai-embeddings";
const TAG_IMAGE: &str = "@muna/test-openai-image";
const TAG_BATCH_SEQUENTIAL: &str = "@muna/test-batch-sequential";
const TAG_BATCH_STATIC: &str = "@muna/test-batch-static";
const TAG_BATCH_DYNAMIC: &str = "@muna/test-batch-dynamic";
const TAG_BATCH_CONTINUOUS: &str = "@muna/test-batch-continuous";
const TAG_SLOW_COLDSTART: &str = "@muna/test-slow-coldstart";

/// First load of a tag downloads the compiled binary; keep generous.
const LOAD_TIMEOUT: Duration = Duration::from_secs(120);

// ---------------------------------------------------------------------------
// Skip plumbing: a missing fake predictor must not fail the whole tier.
// ---------------------------------------------------------------------------

macro_rules! skip_without_access_key {
    () => {
        if !common::access_key_available() {
            eprintln!(
                "SKIPPED: no MUNA_ACCESS_KEY configured; add it to \
                 muna-server/.env (see .env.example)"
            );
            return;
        }
    };
}

/// Skip when the fake predictor has not been pushed. An indeterminate probe
/// (API/network failure) does NOT skip: the test proceeds and surfaces the
/// real error instead of silently hollowing out the suite.
macro_rules! skip_if_unpushed {
    ($tag:expr) => {
        if common::predictor_available($tag).await == Some(false) {
            eprintln!(
                "SKIPPED: {} is not pushed; run the compile commands in \
                 tests/predictors/README.md",
                $tag
            );
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn boot_health_status_heartbeats() {
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();

    // Boot: health + empty status.
    let status: Value = client
        .get(format!("{}/status", server.url()))
        .send().await.unwrap()
        .json().await.unwrap();
    assert_eq!(status["draining"], json!(false));
    assert_eq!(status["models"], json!([]));
    assert!(status["version"].as_str().is_some_and(|v| !v.is_empty()));
    assert!(status["uptime_s"].is_u64());
    assert!(status["gpus"].is_array());
    // Disk capacity on the data volume (statvfs works on macOS + Linux).
    assert!(status["disk_free_mb"].is_u64());
    assert!(status["disk_total_mb"].is_u64());

    // Heartbeats arrive at ~1s cadence carrying the same payload.
    wait_for(Duration::from_secs(5), || async {
        stub.heartbeat_count() >= 3
    })
    .await
    .expect("expected >= 3 heartbeats within 5s at 1s cadence");
    let beat = stub.last_heartbeat().unwrap();
    assert_eq!(beat["node_id"], json!("test-node"));
    assert_eq!(beat["version"], status["version"]);
    assert!(beat["uptime_s"].is_u64());
    assert!(beat["gpus"].is_array());
    assert!(beat["models"].is_array());
}

#[tokio::test(flavor = "multi_thread")]
async fn drain_directive_and_endpoint() {
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let status_url = format!("{}/status", server.url());

    // Directive drain: reconciled on the next beat.
    stub.state().directives.drain = true;
    wait_for(Duration::from_secs(5), || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["draining"] == json!(true)
        }
    })
    .await
    .expect("drain directive did not take effect");

    // Draining rejects inference with 503 before any registry work.
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({ "model": TAG_CHAT, "messages": [] }))
        .send().await.unwrap();
    assert_eq!(response.status(), 503);
    assert!(response.headers().contains_key("Retry-After"));

    // Un-drain restores service.
    stub.state().directives.drain = false;
    wait_for(Duration::from_secs(5), || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["draining"] == json!(false)
        }
    })
    .await
    .expect("un-drain did not take effect");

    // The local /drain endpoint flips the same flag.
    let response = client
        .post(format!("{}/drain", server.url()))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    let status: Value = client.get(&status_url).send().await.unwrap().json().await.unwrap();
    assert_eq!(status["draining"], json!(true));
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_load_reported_then_cleared_by_unload() {
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let status_url = format!("{}/status", server.url());
    let bogus = "@muna/does-not-exist-xyz";

    stub.state().directives.load_models = vec![bogus.to_string()];
    wait_for(Duration::from_secs(30), || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().iter().any(|m| {
                m["tag"] == json!(bogus)
                    && m["state"] == json!("failed")
                    && m["error"].as_str().is_some_and(|e| !e.is_empty())
            })
        }
    })
    .await
    .expect("bogus tag never reported as failed");

    // Unload clears the failed slot.
    stub.state().directives.load_models = vec![];
    stub.state().directives.unload_models = vec![bogus.to_string()];
    wait_for(Duration::from_secs(10), || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().is_empty()
        }
    })
    .await
    .expect("failed model was not cleared by unload");
}

#[tokio::test(flavor = "multi_thread")]
async fn loading_window_returns_429_with_retry_after() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_SLOW_COLDSTART);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let status_url = format!("{}/status", server.url());

    // Warm via directive (non-blocking), then catch the loading window.
    stub.state().directives.load_models = vec![TAG_SLOW_COLDSTART.to_string()];
    wait_for(Duration::from_secs(10), || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().iter().any(|m| m["tag"] == json!(TAG_SLOW_COLDSTART))
        }
    })
    .await
    .expect("slow-coldstart load never started");

    // The predictor is parameterless, so the registry's warmup sentinel runs
    // the full body: the ~12s sleep executes during load, past the registry's
    // 10s hold threshold. A request that arrives mid-load must give up with
    // 429 + Retry-After. The dummy input keeps the inputs map non-empty
    // (empty inputs would take muna's raw-prediction path if timing slipped
    // past the load); the runtime ignores unknown inputs.
    let response = client
        .post(format!("{}/v1/predictions/remote", server.url()))
        .json(&json!({
            "tag": TAG_SLOW_COLDSTART,
            "inputs": { "value": remote_string("hello") },
        }))
        .send().await.unwrap();
    let response_status = response.status();
    let retry_after = response.headers().get("Retry-After").cloned();
    let body = response.text().await.unwrap();
    if response_status == 500 && body.to_lowercase().contains("failed to load") {
        panic!("slow-coldstart load failed instead of loading: {body}");
    }
    assert_eq!(response_status, 429, "expected 429 mid-load, got {response_status}: {body}");
    let retry_after = retry_after.expect("429 must carry Retry-After");
    assert!(retry_after.to_str().unwrap().parse::<u64>().unwrap() >= 1);

    // Eventually ready.
    wait_for(LOAD_TIMEOUT, || {
        let client = client.clone();
        let url = status_url.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().iter().any(|m| {
                m["tag"] == json!(TAG_SLOW_COLDSTART) && m["state"] == json!("ready")
            })
        }
    })
    .await
    .expect("slow-coldstart never became ready");
}

// ---------------------------------------------------------------------------
// OpenAI surface (fake LLM)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn chat_completion_streaming_and_unload() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_CHAT);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let expected_text = "the quick brown fox";
    let request = json!({
        "model": TAG_CHAT,
        "messages": [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": expected_text },
        ],
    });

    // Non-streaming: merged ChatCompletion with usage plumbed through.
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&request)
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap();
    let response_status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(response_status, 200, "chat completion failed: {body}");
    let completion: Value = serde_json::from_str(&body).unwrap();
    assert!(completion["id"].as_str().is_some_and(|id| !id.is_empty()));
    assert_eq!(
        completion["choices"][0]["message"]["content"],
        json!(expected_text),
        "fake chat model echoes the last user message"
    );
    // The fake reports cached_tokens == number of user messages (1 here);
    // a non-null value proves the plumbing through chunk-merge + serialization.
    assert_eq!(
        completion["usage"]["prompt_tokens_details"]["cached_tokens"],
        json!(1)
    );
    assert!(completion["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    // The fake emits a reasoning delta before its content deltas; the merged
    // message must surface it (DeepSeek `reasoning_content` convention),
    // along with the reasoning token count in usage.
    assert_eq!(
        completion["choices"][0]["message"]["reasoning_content"],
        json!("thinking really hard")
    );
    assert_eq!(
        completion["usage"]["completion_tokens_details"]["reasoning_tokens"],
        json!(3)
    );

    // Ready model shows up in /v1/models and /status.
    let models: Value = client
        .get(format!("{}/v1/models", server.url()))
        .send().await.unwrap().json().await.unwrap();
    assert!(models["data"].as_array().unwrap().iter().any(|m| m["id"] == json!(TAG_CHAT)));

    // Streaming: multiple SSE frames, then [DONE]; chunks reassemble the text.
    let mut streaming_request = request.clone();
    streaming_request["stream"] = json!(true);
    let sse = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&streaming_request)
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap()
        .text().await.unwrap();
    let frames: Vec<&str> = sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert!(frames.len() >= 3, "expected several SSE frames, got: {sse}");
    assert_eq!(*frames.last().unwrap(), "[DONE]");
    let mut streamed_text = String::new();
    let mut streamed_reasoning = String::new();
    for frame in &frames[..frames.len() - 1] {
        let chunk: Value = serde_json::from_str(frame).unwrap();
        let delta = &chunk["choices"][0]["delta"];
        // Deltas carry `reasoning_content` XOR `content` (DeepSeek clients
        // break when both appear in one chunk).
        assert!(
            !(
                delta["content"].as_str().is_some_and(|c| !c.is_empty()) &&
                delta["reasoning_content"].as_str().is_some()
            ),
            "delta carries both content and reasoning_content: {delta}"
        );
        if let Some(content) = delta["content"].as_str() {
            streamed_text.push_str(content);
        }
        if let Some(reasoning) = delta["reasoning_content"].as_str() {
            streamed_reasoning.push_str(reasoning);
        }
    }
    assert_eq!(streamed_text, expected_text);
    assert_eq!(streamed_reasoning, "thinking really hard");

    // Unload directive removes the model from /v1/models and /status.
    stub.state().directives.unload_models = vec![TAG_CHAT.to_string()];
    wait_for(Duration::from_secs(10), || {
        let client = client.clone();
        let url = format!("{}/v1/models", server.url());
        async move {
            let models: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            models["data"].as_array().unwrap().is_empty()
        }
    })
    .await
    .expect("unload directive did not remove the model");
}

#[tokio::test(flavor = "multi_thread")]
async fn embeddings_shape_determinism_usage() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_EMBEDDINGS);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let request = json!({
        "model": TAG_EMBEDDINGS,
        "input": ["What is the capital of France?", "Butterflies have legs."],
        "dimensions": 64,
    });

    let response = client
        .post(format!("{}/v1/embeddings", server.url()))
        .json(&request)
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap();
    let response_status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(response_status, 200, "embeddings failed: {body}");
    let first: Value = serde_json::from_str(&body).unwrap();
    let data = first["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    for (i, embedding) in data.iter().enumerate() {
        assert_eq!(embedding["index"], json!(i));
        assert_eq!(embedding["embedding"].as_array().unwrap().len(), 64);
    }
    // The fake reports one token per whitespace-separated word: 6 + 3.
    assert_eq!(first["usage"]["prompt_tokens"], json!(9));

    // Hash-derived embeddings: a second call returns identical vectors.
    let second: Value = client
        .post(format!("{}/v1/embeddings", server.url()))
        .json(&request)
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap()
        .json().await.unwrap();
    assert_eq!(first["data"], second["data"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn image_generations_b64_png() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_IMAGE);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{}/v1/images/generations", server.url()))
        .json(&json!({
            "model": TAG_IMAGE,
            "prompt": "a photo of a cat",
            "size": "256x256",
        }))
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap();
    let response_status = response.status();
    let body = response.text().await.unwrap();
    assert_eq!(response_status, 200, "image generation failed: {body}");
    let generated: Value = serde_json::from_str(&body).unwrap();
    let data = generated["data"].as_array().unwrap();
    assert_eq!(data.len(), 1, "one image per prompt");
    let b64 = data[0]["b64_json"].as_str().expect("expected b64_json image");
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "default output format is png");
}

// ---------------------------------------------------------------------------
// Dispatcher over HTTP (/v1/predictions/remote)
// ---------------------------------------------------------------------------

async fn remote_prediction(
    client: &reqwest::Client,
    server_url: &str,
    body: Value,
) -> (reqwest::StatusCode, String) {
    let response = client
        .post(format!("{server_url}/v1/predictions/remote"))
        .json(&body)
        .timeout(LOAD_TIMEOUT)
        .send().await.unwrap();
    let status = response.status();
    let text = response.text().await.unwrap();
    (status, text)
}

/// Extract the first result of a RemotePrediction response as JSON.
fn first_result(prediction_body: &str) -> Value {
    let prediction: Value = serde_json::from_str(prediction_body).unwrap();
    assert!(
        prediction["error"].is_null(),
        "prediction failed: {}", prediction["error"]
    );
    decode_remote_json(&prediction["results"][0])
}

fn window_of(result: &Value) -> (f64, f64) {
    (result["start"].as_f64().unwrap(), result["end"].as_f64().unwrap())
}

fn windows_overlap(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

#[tokio::test(flavor = "multi_thread")]
async fn sequential_dispatch_serializes_requests() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_BATCH_SEQUENTIAL);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let server_url = server.url();
    let request = |value: &str| json!({
        "tag": TAG_BATCH_SEQUENTIAL,
        "inputs": { "value": remote_string(value) },
    });

    // Warm up so the batch under test is not skewed by a cold load.
    let (_, body) = remote_prediction(&client, &server_url, request("warmup")).await;
    let _ = first_result(&body);

    // Two concurrent requests: the per-model mutex must serialize the
    // predictor's ~400ms sleeps into disjoint [start, end] windows.
    let (a, b) = tokio::join!(
        remote_prediction(&client, &server_url, request("a")),
        remote_prediction(&client, &server_url, request("b")),
    );
    let result_a = first_result(&a.1);
    let result_b = first_result(&b.1);
    assert_eq!(result_a["value"], json!("a"));
    assert_eq!(result_b["value"], json!("b"));
    let window_a = window_of(&result_a);
    let window_b = window_of(&result_b);
    assert!(
        !windows_overlap(window_a, window_b),
        "sequential dispatch must not overlap: {window_a:?} vs {window_b:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_static_merges_full_batch() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_BATCH_STATIC);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let server_url = server.url();
    let request = |item: &str| json!({
        "tag": TAG_BATCH_STATIC,
        "inputs": { "items": remote_list(json!([item])) },
    });

    // Warm up so the batch under test is not skewed by a cold load.
    let (_, body) = remote_prediction(&client, &server_url, request("warmup")).await;
    let _ = first_result(&body);

    // Four one-item requests fill capacity 4: one merged invocation whose
    // shared timestamps come back identical, split into per-caller items.
    let (r0, r1, r2, r3) = tokio::join!(
        remote_prediction(&client, &server_url, request("i0")),
        remote_prediction(&client, &server_url, request("i1")),
        remote_prediction(&client, &server_url, request("i2")),
        remote_prediction(&client, &server_url, request("i3")),
    );
    let results: Vec<Value> = [&r0.1, &r1.1, &r2.1, &r3.1]
        .iter()
        .map(|body| first_result(body))
        .collect();
    for (i, result) in results.iter().enumerate() {
        let items = result.as_array().expect("split returns the caller's slice");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["item"], json!(format!("i{i}")));
    }
    let windows: Vec<(f64, f64)> = results
        .iter()
        .map(|r| window_of(&r.as_array().unwrap()[0]))
        .collect();
    assert!(
        windows.iter().all(|w| *w == windows[0]),
        "merged batch must share one invocation's timestamps: {windows:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn buffered_dynamic_holds_mismatched_key() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_BATCH_DYNAMIC);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let server_url = server.url();
    let request = |item: &str, prefix: &str| json!({
        "tag": TAG_BATCH_DYNAMIC,
        "inputs": {
            "items": remote_list(json!([item])),
            "prefix": remote_string(prefix),
        },
    });

    // Warm up so the batch under test is not skewed by a cold load.
    let (_, body) = remote_prediction(&client, &server_url, request("warmup", "w:")).await;
    let _ = first_result(&body);

    // Same-key requests merge; the mismatched-prefix request is held for
    // its own invocation (different broadcast params must never merge).
    let (same_1, same_2, mismatched) = tokio::join!(
        remote_prediction(&client, &server_url, request("a", "x:")),
        remote_prediction(&client, &server_url, request("b", "x:")),
        remote_prediction(&client, &server_url, request("c", "y:")),
    );
    let result_1 = first_result(&same_1.1);
    let result_2 = first_result(&same_2.1);
    let result_3 = first_result(&mismatched.1);
    assert_eq!(result_1.as_array().unwrap()[0]["item"], json!("x:a"));
    assert_eq!(result_2.as_array().unwrap()[0]["item"], json!("x:b"));
    assert_eq!(result_3.as_array().unwrap()[0]["item"], json!("y:c"));
    let window_1 = window_of(&result_1.as_array().unwrap()[0]);
    let window_2 = window_of(&result_2.as_array().unwrap()[0]);
    let window_3 = window_of(&result_3.as_array().unwrap()[0]);
    assert_eq!(window_1, window_2, "same-key requests share one invocation");
    assert_ne!(window_1, window_3, "mismatched key lands in a separate invocation");
}

#[tokio::test(flavor = "multi_thread")]
async fn continuous_dispatch_overlaps() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_BATCH_CONTINUOUS);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    let server_url = server.url();
    let request = |item: &str| json!({
        "tag": TAG_BATCH_CONTINUOUS,
        "inputs": { "items": remote_list(json!([item])) },
    });

    // Warm up so the batch under test is not skewed by a cold load.
    let (_, body) = remote_prediction(&client, &server_url, request("warmup")).await;
    let _ = first_result(&body);

    // Continuous mode has no lock and no buffering: the two ~400ms sleeps
    // must run concurrently, i.e. the windows overlap.
    let (a, b) = tokio::join!(
        remote_prediction(&client, &server_url, request("a")),
        remote_prediction(&client, &server_url, request("b")),
    );
    let result_a = first_result(&a.1);
    let result_b = first_result(&b.1);
    let window_a = window_of(&result_a.as_array().unwrap()[0]);
    let window_b = window_of(&result_b.as_array().unwrap()[0]);
    assert!(
        windows_overlap(window_a, window_b),
        "continuous dispatch must overlap: {window_a:?} vs {window_b:?}"
    );
}

// ---------------------------------------------------------------------------
// Pinned model set (--models)
// ---------------------------------------------------------------------------

/// Finds `tag` in a status payload's `models` array.
fn model_in<'a>(status: &'a Value, tag: &str) -> Option<&'a Value> {
    status["models"].as_array().unwrap().iter().find(|m| m["tag"] == json!(tag))
}

#[tokio::test(flavor = "multi_thread")]
async fn pinned_models_eager_load_and_reject_unlisted() {
    skip_without_access_key!();
    skip_if_unpushed!(TAG_BATCH_SEQUENTIAL);
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn_with_models(&stub, &[TAG_BATCH_SEQUENTIAL]).await;
    let client = reqwest::Client::new();
    let status_url = format!("{}/status", server.url());

    // Eager load: the pinned tag reaches `ready` with no request and no
    // control-plane directive.
    wait_for(LOAD_TIMEOUT, || async {
        let status: Value = client
            .get(&status_url).send().await.unwrap()
            .json().await.unwrap();
        model_in(&status, TAG_BATCH_SEQUENTIAL)
            .is_some_and(|m| m["state"] == json!("ready"))
    })
    .await
    .expect("pinned tag never eager-loaded to ready");

    // An unlisted tag is rejected with OpenAI's model_not_found shape,
    // without touching the registry (immediate, not a doomed load).
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&json!({ "model": TAG_CHAT, "messages": [{ "role": "user", "content": "hi" }] }))
        .send().await.unwrap();
    assert_eq!(response.status(), 404);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["error"]["code"], json!("model_not_found"));

    // A control-plane warm directive for an unlisted tag is ignored: the
    // tag never appears in status.
    stub.state().directives.load_models = vec![TAG_CHAT.to_string()];
    tokio::time::sleep(Duration::from_secs(3)).await;
    let status: Value = client
        .get(&status_url).send().await.unwrap()
        .json().await.unwrap();
    assert!(model_in(&status, TAG_CHAT).is_none());
}
