/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Control-plane heartbeat loop.
//!
//! Every `heartbeat_interval`, POST the full `NodeStatus` (per-model state,
//! GPU metrics) to `{control}/v1/nodes/{node_id}/heartbeat`. The response
//! is a declarative list of goal descriptors (tag, residency `process` |
//! `disk` | `none`, optional download key); the node diffs it against its
//! own actual state and walks the warmth ladder locally. An absent tag
//! means "no opinion", so an empty response (also the parse-failure
//! default) is a guaranteed no-op.

use std::sync::Arc;

use crate::control::protocol::{HeartbeatResponse, NodeStatus, Residency};
use crate::state::AppState;

pub(crate) async fn run(state: Arc<AppState>) {
    let node = state.node.as_ref().expect("heartbeat requires node context");
    let url = format!(
        "{}/v1/nodes/{}/heartbeat",
        node.control_plane_url.trim_end_matches('/'),
        node.node_id
    );
    let token = std::env::var("MUNA_SERVER_TOKEN").ok();
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

/// Diff the goal residency map against local state. Every arm is
/// idempotent, so re-applying the same goal each beat is free.
async fn apply(state: &Arc<AppState>, reconcile: HeartbeatResponse) {
    // Store download credentials BEFORE acting on residency: the registry
    // loader and cache downloader read the key store when they build the
    // per-model Muna clients the directives below trigger.
    for descriptor in &reconcile.models {
        if let Some(key) = &descriptor.key {
            state.keys.insert(descriptor.tag.clone(), key.clone());
        }
    }
    for descriptor in &reconcile.models {
        let tag = &descriptor.tag;
        // A residency goal for a tag outside the pinned set (`--models`) is
        // a control-plane misconfiguration: neither loaded nor cached.
        if !state.registry.serves(tag) {
            if !matches!(descriptor.residency, Residency::None) {
                tracing::warn!(
                    tag = %tag,
                    "residency goal ignored: tag is not in this server's --models set"
                );
            }
            continue;
        }
        match descriptor.residency {
            Residency::Process => {
                // `process` implies `disk`: track the cached tier too, so a
                // later demotion reports `cached` instead of vanishing.
                // The engine load downloads the same resources; the
                // client's per-path single-flight de-duplicates the work.
                state.cache.ensure_cached(tag);
                state.registry.warm_reconcile(tag);
            }
            Residency::Disk => {
                // Demote: engine out (idempotent no-op when not loaded),
                // resources on disk.
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                state.cache.ensure_cached(tag);
            }
            Residency::None => {
                // Engine out. Disk eviction is PERMITTED but not required;
                // genuinely cached resources stay (and keep reporting
                // `cached`) until node-local GC under disk pressure exists.
                // A failed cache record, by contrast, is forgotten: nothing
                // is on disk, so there is nothing to keep reporting.
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                state.cache.forget_failed(tag);
            }
        }
    }
    if reconcile.drain != state.is_draining() {
        tracing::info!(drain = reconcile.drain, "drain state changed by control plane");
        state.set_draining(reconcile.drain);
    }
}
