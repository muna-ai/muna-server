/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Control-plane heartbeat loop.
//!
//! Every `heartbeat_interval` or immediately when reported state changes
//! (`NotificationCenter::status`), POST the full `NodeStatus` (per-model
//! state, GPU metrics) to `{control}/v1/nodes/{node_id}/heartbeat`. The
//! response is a declarative list of goal descriptors (tag, residency
//! `process` | `disk` | `none`, optional download key); the node diffs it
//! against its own actual state and walks the warmth ladder locally. An
//! absent tag means "no opinion", so an empty response (also the
//! parse-failure default) is a guaranteed no-op.

use std::sync::Arc;

use crate::control::protocol::{HeartbeatResponse, ModelDescriptor, NodeStatus, Residency};
use crate::serving::cache::CachePriority;
use crate::state::AppState;

pub(crate) async fn run(state: Arc<AppState>) {
    let node = state.mode.node().expect("heartbeat requires node context");
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
        // Periodic cadence OR a state transition. `Notify::notify_one`
        // stores at most one permit, so any number of transitions that
        // land while a beat is in flight collapse into exactly one
        // follow-up beat; no explicit coalescing needed.
        tokio::select! {
            _ = state.shutdown.cancelled() => break,
            _ = interval.tick() => {}
            _ = state.notifications.status.notified() => {}
        }
        beat(&state, &client, &url, token.as_deref()).await;
    }
    // One last report (`draining: true`) so the plane stops routing here
    // now rather than after the heartbeat timeout.
    beat(&state, &client, &url, token.as_deref()).await;
}

/// One heartbeat: report status, apply the goals that come back.
async fn beat(
    state: &Arc<AppState>,
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>
) {
    let payload = NodeStatus::collect(state);
    let mut request = client.post(url).json(&payload);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = match request.send().await {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(error = %e, "heartbeat failed");
            return;
        }
    };
    if !response.status().is_success() {
        tracing::warn!(status = %response.status(), "heartbeat rejected");
        return;
    }
    let reconcile: HeartbeatResponse = match response.json().await {
        Ok(reconcile) => reconcile,
        Err(e) => {
            tracing::warn!(error = %e, "malformed heartbeat response");
            return;
        }
    };
    // Once shutdown starts, goals would start loads that cannot finish
    // and could un-drain the node.
    if state.shutdown.is_cancelled() {
        return;
    }
    apply(state, reconcile).await;
}

/// Diff the goal residency map against local state. Every arm is
/// idempotent, so re-applying the same goal each beat is free.
async fn apply(state: &Arc<AppState>, reconcile: HeartbeatResponse) {
    state.lease.retain_reservations(|tag| state.registry.devices(tag).is_some());
    let mut rejected = std::collections::HashSet::new();
    for descriptor in &reconcile.models {
        if !matches!(descriptor.residency, Residency::Process) || !state.registry.serves(&descriptor.tag) {
            continue;
        }
        let Some(devices) = descriptor.devices.as_deref().filter(|ids| !ids.is_empty() && valid_devices(Some(ids))) else { continue; };
        let dedicated = descriptor.policy.is_some_and(|policy| policy.dedicated);
        let current = state.registry.devices(&descriptor.tag);
        let resident = current.is_some();
        // Retire a moved assignment first; its old reservation cannot be
        // replaced until native work and the old engine have drained.
        if current.as_ref().is_some_and(|ids| ids.as_deref() != Some(devices)) {
            continue;
        }
        // Validate existing peers before loading the dedicated owner. Unknown
        // loading engines are not evidence of cooperative lease support.
        let incompatible_peer = dedicated && state.registry.snapshot().iter().any(|(tag, model)| {
            if tag == &descriptor.tag { return false; }
            match model {
                crate::serving::registry::ModelState::Loading { devices: Some(ids), .. } => ids.iter().any(|id| devices.contains(id)),
                crate::serving::registry::ModelState::Ready(model) => model.devices.as_ref().is_some_and(|ids|
                    ids.iter().any(|id| devices.contains(id)) && !state.lease.supports(tag, ids)),
                _ => false,
            }
        });
        // Unverified new engines must not warm up on a live dedicated GPU.
        // Preload shared peers before admitting its dedicated owner.
        if incompatible_peer || (!dedicated && !resident && state.lease.protected(devices)) ||
            (dedicated && !state.lease.reserve_dedicated(&descriptor.tag, devices)) {
            tracing::error!(tag = %descriptor.tag, "process goal rejected: dedicated reservation or unsupported lease peer");
            rejected.insert(descriptor.tag.clone());
        }
    }
    // Recheck after all dedicated reservations are known; descriptor order
    // must not admit an unverified new shared engine in the same heartbeat.
    for descriptor in &reconcile.models {
        if matches!(descriptor.residency, Residency::Process) &&
            !descriptor.policy.is_some_and(|policy| policy.dedicated) &&
            state.registry.devices(&descriptor.tag).is_none() &&
            descriptor.devices.as_ref().is_some_and(|ids| state.lease.protected(ids)) {
            rejected.insert(descriptor.tag.clone());
        }
    }
    // Store download credentials and device assignments BEFORE acting on
    // residency: the registry loader and cache downloader read them when
    // the directives below trigger a load. Devices and policy mirror the
    // descriptor exactly (absent clears), since both are per-beat goals.
    for descriptor in &reconcile.models {
        let tag = &descriptor.tag;
        if let Some(key) = &descriptor.key {
            state.keys.insert(tag.clone(), key.clone());
        }
        if let Some(devices) = &descriptor.devices {
            state.devices.insert(tag.clone(), devices.clone());
        } else {
            state.devices.remove(tag);
        }
        if let Some(policy) = descriptor.policy {
            state.policies.insert(tag.clone(), policy);
        } else {
            state.policies.remove(tag);
        }
    }
    for descriptor in process_first(&reconcile.models) {
        let tag = &descriptor.tag;
        if rejected.contains(tag) {
            continue;
        }
        if let Some(model) = state.registry.ready(tag) {
            let dedicated = descriptor.policy.is_some_and(|policy| policy.dedicated);
            if model.lease.as_ref().is_some_and(|lease| lease.dedicated != dedicated) {
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                continue;
            }
        }
        if matches!(descriptor.residency, Residency::Process) && !valid_devices(descriptor.devices.as_deref()) {
            tracing::error!(tag = %tag, "process goal requires an explicit device list with distinct valid GPU IDs (empty for CPU)");
            state.dispatcher.remove(tag);
            state.registry.unload(tag).await;
            continue;
        }
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
                // One predictor per process, so moving a model to other
                // devices means unload then reload. A `Loading` engine
                // defers the unload to its load task; the slot then clears
                // and the next beat reloads on the new devices.
                if let Some(current) = state.registry.devices(tag) {
                    if current != descriptor.devices {
                        tracing::info!(
                            tag = %tag,
                            from = ?current,
                            to = ?descriptor.devices,
                            "device assignment changed; reloading"
                        );
                        state.dispatcher.remove(tag);
                        state.registry.unload(tag).await;
                        continue;
                    }
                }
                // Warm FIRST so the gate guard exists before the cache
                // request. `process` implies `disk`: track the cached tier
                // too, so a later demotion reports `cached` instead of
                // vanishing. The engine load downloads the same resources
                // and the client's per-path single-flight de-duplicates,
                // so the cache request is `Immediate` rather than queued.
                state.registry.warm_reconcile(tag);
                state.cache.ensure_cached(tag, CachePriority::Immediate);
            }
            Residency::Disk => {
                // Demote: engine out (idempotent no-op when not loaded),
                // resources on disk. Prefetch queues behind process-tier
                // downloads.
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                state.cache.ensure_cached(tag, CachePriority::Prefetch);
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

fn valid_devices(devices: Option<&[u32]>) -> bool {
    let Some(devices) = devices else { return false; };
    devices.iter().all(|id| *id <= i32::MAX as u32) &&
        devices.iter().copied().collect::<std::collections::HashSet<_>>().len() == devices.len()
}

/// Order directives so `process` goals are applied before the rest.
///
/// `warm` takes the download-gate guard synchronously, so every disk-tier
/// prefetch applied after the `process` pass queues behind the loads the
/// plane is actually waiting on, whatever order the plane listed them in.
/// Stable within each group.
fn process_first(models: &[ModelDescriptor]) -> impl Iterator<Item = &ModelDescriptor> {
    let (process, rest): (Vec<_>, Vec<_>) = models
        .iter()
        .partition(|descriptor| matches!(descriptor.residency, Residency::Process));
    process.into_iter().chain(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(tag: &str, residency: Residency) -> ModelDescriptor {
        ModelDescriptor {
            tag: tag.to_string(),
            residency,
            key: None,
            devices: None,
            policy: None
        }
    }

    #[test]
    fn process_directives_are_applied_first() {
        // The plane listed a disk-tier prefetch ahead of the process-tier
        // load; the load must still be applied first so its gate guard
        // exists before the prefetch is spawned.
        let models = vec![
            descriptor("@a/flux", Residency::Disk),
            descriptor("@a/gemma", Residency::Process),
            descriptor("@a/old", Residency::None),
            descriptor("@a/qwen", Residency::Process),
        ];
        let tags: Vec<&str> = process_first(&models).map(|d| d.tag.as_str()).collect();
        assert_eq!(tags, vec!["@a/gemma", "@a/qwen", "@a/flux", "@a/old"]);
    }
    #[test]
    fn process_devices_must_be_explicit_and_distinct() {
        assert!(!valid_devices(None));
        assert!(valid_devices(Some(&[])));
        assert!(!valid_devices(Some(&[0, 0])));
        assert!(!valid_devices(Some(&[u32::MAX])));
        assert!(valid_devices(Some(&[2, 0])));
    }

}
