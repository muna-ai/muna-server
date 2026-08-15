/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Control-plane heartbeat loop.
//!
//! Every `heartbeat_interval`, POST the full `NodeStatus` (per-model state,
//! GPU metrics) to `{control}/v1/nodes/{node_id}/heartbeat`. The response
//! drives reconciliation with imperative deltas: `load_models` /
//! `unload_models` are idempotent no-ops for already-satisfied tags, and the
//! control plane self-heals by diffing against the fresh status every beat.

use std::sync::Arc;

use crate::control::protocol::{HeartbeatResponse, NodeStatus};
use crate::state::AppState;

pub(crate) async fn run(state: Arc<AppState>) {
    let node = state.node.as_ref().expect("heartbeat requires node context");
    let url = format!(
        "{}/v1/nodes/{}/heartbeat",
        node.control_plane_url.trim_end_matches('/'),
        node.node_id
    );
    let token = std::env::var("MUNA_NODE_TOKEN").ok();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build heartbeat client");
    let mut interval = tokio::time::interval(node.heartbeat_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        let payload = NodeStatus::collect(&state);
        let mut request = client.post(&url).json(&payload);
        if let Some(token) = &token {
            request = request.bearer_auth(token);
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => {
                tracing::warn!(error = %e, "heartbeat failed");
                continue;
            }
        };
        if !response.status().is_success() {
            tracing::warn!(status = %response.status(), "heartbeat rejected");
            continue;
        }
        let reconcile: HeartbeatResponse = match response.json().await {
            Ok(reconcile) => reconcile,
            Err(e) => {
                tracing::warn!(error = %e, "malformed heartbeat response");
                continue;
            }
        };
        apply(&state, reconcile).await;
    }
}

async fn apply(state: &Arc<AppState>, reconcile: HeartbeatResponse) {
    for tag in &reconcile.load_models {
        state.registry.warm(tag);
    }
    for tag in &reconcile.unload_models {
        state.dispatcher.remove(tag);
        state.registry.unload(tag).await;
    }
    if let Some(node) = &state.node {
        node.event_callbacks.send_if_modified(|current| {
            if *current == reconcile.event_callback_urls {
                false
            } else {
                *current = reconcile.event_callback_urls.clone();
                true
            }
        });
    }
    if reconcile.drain != state.is_draining() {
        tracing::info!(drain = reconcile.drain, "drain state changed by control plane");
        state.set_draining(reconcile.drain);
    }
}
