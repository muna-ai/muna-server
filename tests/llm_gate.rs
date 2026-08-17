/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! LLM gate: real-engine serving behavior (`#[ignore]`, CUDA box only).
//!
//! Everything here genuinely needs the engine - the real loading window,
//! prefix-cache `cached_tokens`, engine-internal continuous batching, and
//! the KV relay handshake - driven against the GLM zero-fill tag.
//! Run with:
//! `MUNA_LLM_GATE_TAG=<tag> cargo test --test llm_gate -- --ignored`
//!
//! The signature-driven local tier lives in `tests/serving.rs`.

mod common;

use std::time::Duration;

use serde_json::{json, Value};

use common::{wait_for, ServerGuard, StubControlPlane};

// ---------------------------------------------------------------------------
// GPU tier: real engine behavior on the GLM zero-fill tag.
// Run: MUNA_LLM_GATE_TAG=<tag> cargo test --test llm_gate -- --ignored
// ---------------------------------------------------------------------------

fn gate_tag() -> Option<String> {
    match std::env::var("MUNA_LLM_GATE_TAG") {
        Ok(tag) if !tag.is_empty() => Some(tag),
        _ => {
            eprintln!("SKIPPED: set MUNA_LLM_GATE_TAG=<glm-zero-fill-tag> to run GPU tier");
            None
        }
    }
}

/// Real engine loads take minutes (weights staging + CUDA graph capture).
const GPU_LOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);

async fn wait_until_ready(client: &reqwest::Client, server_url: &str, tag: &str) {
    let status_url = format!("{server_url}/status");
    wait_for(GPU_LOAD_TIMEOUT, || {
        let client = client.clone();
        let url = status_url.clone();
        let tag = tag.to_string();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().iter().any(|m| {
                if m["tag"] == json!(tag) && m["state"] == json!("failed") {
                    panic!("model load failed: {}", m["error"]);
                }
                m["tag"] == json!(tag) && m["state"] == json!("ready")
            })
        }
    })
    .await
    .expect("engine never became ready");
}

fn chat_request(tag: &str, prompt: &str) -> Value {
    json!({
        "model": tag,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": 32,
    })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn gpu_real_load_chat_and_cached_tokens() {
    let Some(tag) = gate_tag() else { return };
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();

    // Real loading window: the engine takes minutes; a chat that arrives
    // mid-load must 503 with Retry-After.
    stub.state().directives.load_models = vec![tag.clone()];
    wait_for(Duration::from_secs(30), || {
        let client = client.clone();
        let url = format!("{}/status", server.url());
        let tag = tag.clone();
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().iter().any(|m| m["tag"] == json!(tag))
        }
    })
    .await
    .expect("load never started");
    let mid_load = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, "hello"))
        .send().await.unwrap();
    assert_eq!(mid_load.status(), 503, "chat mid-load must 503");
    assert!(mid_load.headers().contains_key("Retry-After"));

    wait_until_ready(&client, &server.url(), &tag).await;

    // Real cached_tokens: a long shared prefix, twice. The second response
    // must report prefix-cache hits.
    let long_prompt = "Repeat this carefully. ".repeat(64);
    let first: Value = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, &long_prompt))
        .timeout(Duration::from_secs(120))
        .send().await.unwrap()
        .json().await.unwrap();
    assert!(first["usage"]["prompt_tokens"].as_i64().unwrap() > 0);
    // Zero-fill guarantee: garbage but valid output, no NaN poisoning.
    assert!(first["choices"][0]["message"]["content"].is_string());
    let second: Value = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, &long_prompt))
        .timeout(Duration::from_secs(120))
        .send().await.unwrap()
        .json().await.unwrap();
    let cached = second["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_i64()
        .expect("second response must carry cached_tokens");
    assert!(cached > 0, "expected prefix-cache hits on the second identical prompt");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn gpu_concurrent_chats_overlap() {
    let Some(tag) = gate_tag() else { return };
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    stub.state().directives.load_models = vec![tag.clone()];
    wait_until_ready(&client, &server.url(), &tag).await;

    // Baseline single-request latency.
    let single_start = std::time::Instant::now();
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, "warmup prompt"))
        .timeout(Duration::from_secs(120))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    let single = single_start.elapsed();

    // N concurrent chats: engine-internal continuous batching must overlap
    // them, so total wall time stays well under N * single.
    const N: usize = 4;
    let concurrent_start = std::time::Instant::now();
    let mut handles = Vec::new();
    for i in 0..N {
        let client = client.clone();
        let url = format!("{}/v1/chat/completions", server.url());
        let body = chat_request(&tag, &format!("prompt number {i}"));
        handles.push(tokio::spawn(async move {
            client.post(&url)
                .json(&body)
                .timeout(Duration::from_secs(300))
                .send().await.unwrap()
                .status()
        }));
    }
    for handle in handles {
        assert_eq!(handle.await.unwrap(), 200);
    }
    let concurrent = concurrent_start.elapsed();
    assert!(
        concurrent < single * (N as u32),
        "no overlap: {N} concurrent chats took {concurrent:?} vs single {single:?}"
    );

    // Counters moved.
    let status: Value = client
        .get(format!("{}/status", server.url()))
        .send().await.unwrap().json().await.unwrap();
    let model = status["models"].as_array().unwrap().iter()
        .find(|m| m["tag"] == json!(tag)).unwrap();
    assert!(model["total_predictions"].as_u64().unwrap() >= (N as u64) + 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn gpu_kv_relay_snapshot_delta_need_snapshot_unload() {
    let Some(tag) = gate_tag() else { return };
    let stub = StubControlPlane::start().await;
    let server = ServerGuard::spawn(&stub).await;
    let client = reqwest::Client::new();
    stub.state().directives.load_models = vec![tag.clone()];
    stub.state().directives.event_callback_urls = vec![stub.kv_callback_url()];
    wait_until_ready(&client, &server.url(), &tag).await;

    // Drive KV traffic so the engine admits blocks.
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, &"KV relay test prompt. ".repeat(32)))
        .timeout(Duration::from_secs(120))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);

    // First batch for this model begins with a snapshot.
    wait_for(Duration::from_secs(60), || async {
        !stub.state().kv_batches.is_empty()
    })
    .await
    .expect("no KV batches relayed");
    {
        let state = stub.state();
        let first = &state.kv_batches[0];
        assert_eq!(first["model"], json!(tag));
        assert_eq!(first["snapshot"], json!(true), "first batch must be a snapshot");
        assert!(first["epoch"].as_str().is_some_and(|e| e.len() == 32));
        assert!(first["worker_id"].as_str().is_some());

        // Contiguous seq ranges under one epoch.
        let epoch = first["epoch"].clone();
        let mut expected_next: Option<u64> = None;
        for batch in state.kv_batches.iter().filter(|b| b["model"] == json!(tag)) {
            assert_eq!(batch["epoch"], epoch, "epoch must be stable within a run");
            let range = batch["seq_range"].as_array().unwrap();
            let (first_seq, last_seq) = (range[0].as_u64().unwrap(), range[1].as_u64().unwrap());
            assert!(first_seq <= last_seq);
            if let Some(expected) = expected_next {
                assert_eq!(first_seq, expected, "seq gap between relayed batches");
            }
            expected_next = Some(last_seq + 1);
        }
    }

    // Armed need_snapshot forces a fresh snapshot on a later batch.
    let batches_before = stub.state().kv_batches.len();
    stub.state().need_snapshot_armed = true;
    let response = client
        .post(format!("{}/v1/chat/completions", server.url()))
        .json(&chat_request(&tag, &"Another prompt for more KV events. ".repeat(32)))
        .timeout(Duration::from_secs(120))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    wait_for(Duration::from_secs(60), || async {
        stub.state().kv_batches[batches_before..]
            .iter()
            .any(|b| b["snapshot"] == json!(true))
    })
    .await
    .expect("need_snapshot did not trigger a fresh snapshot");

    // Unload stops the batches.
    stub.state().directives.load_models = vec![];
    stub.state().directives.unload_models = vec![tag.clone()];
    wait_for(Duration::from_secs(30), || {
        let client = client.clone();
        let url = format!("{}/status", server.url());
        async move {
            let status: Value = client.get(&url).send().await.unwrap().json().await.unwrap();
            status["models"].as_array().unwrap().is_empty()
        }
    })
    .await
    .expect("unload never completed");
    let count_after_unload = stub.state().kv_batches.len();
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(
        stub.state().kv_batches.len(),
        count_after_unload,
        "KV batches must stop after unload"
    );
}
