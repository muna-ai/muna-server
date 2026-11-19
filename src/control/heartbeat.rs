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

use tokio::sync::watch;

use crate::control::protocol::{HeartbeatResponse, HeartbeatVersion, ModelDescriptor, NodeStatus, Residency};
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
    let cache = state.cache_path.clone();
    let session = match tokio::task::spawn_blocking(move || super::session::next(&cache)).await {
        Ok(Ok(session)) => session,
        result => {
            tracing::error!(?result, "cannot allocate heartbeat session; refusing unsequenced reports");
            return;
        }
    };
    let (goals, updates) = watch::channel(None);
    // Goal application can wait for an engine to unload. Keep reporting state
    // throughout that wait; applies remain serialized and are never cancelled.
    tokio::join!(
        report(&state, &client, &url, token.as_deref(), session, goals),
        reconcile(&state, updates)
    );
}

async fn report(
    state: &Arc<AppState>,
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    session: u64,
    goals: watch::Sender<Option<HeartbeatResponse>>
) {
    use super::reporting::{self, Report};

    let mut version = HeartbeatVersion { session, sequence: 0 };
    let mut interval = tokio::time::interval(state.mode.node().unwrap().heartbeat_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    'reporting: loop {
        tokio::select! {
            biased;
            _ = state.shutdown.cancelled() => break,
            _ = state.notifications.status.notified() => {},
            _ = interval.tick() => {},
        }
        loop {
            version.sequence += 1;
            let request = request(state, client, url, token, version);
            match reporting::exchange(request, version, &state.notifications.status, &state.shutdown).await {
                // The previous HTTP request is dropped. Its server-side work
                // may survive, so the plane must also reject stale versions.
                Report::Changed => continue,
                Report::Stopped => break 'reporting,
                Report::Complete(response) => {
                    if let Some(response) = response {
                        goals.send_replace(Some(response));
                    }
                    break;
                }
            }
        }
    }
    // Shutdown has already cancelled the token, so send this final draining
    // snapshot directly. Never apply its response or start new loads.
    version.sequence += 1;
    let _ = reporting::send(request(state, client, url, token, version), version).await;
}

fn request(
    state: &AppState,
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    version: HeartbeatVersion
) -> reqwest::RequestBuilder {
    let mut payload = NodeStatus::collect(state);
    payload.heartbeat = version;
    let mut request = client.post(url).json(&payload);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    request
}

async fn reconcile(
    state: &Arc<AppState>,
    mut updates: watch::Receiver<Option<HeartbeatResponse>>
) {
    loop {
        tokio::select! {
            biased;
            _ = state.shutdown.cancelled() => break,
            changed = updates.changed() => if changed.is_err() { break; },
        }
        // Keep only the newest completed response while a previous apply waits.
        let response = updates.borrow_and_update().clone();
        if let Some(response) = response {
            if state.shutdown.is_cancelled() { break; }
            apply(state, response).await;
        }
    }
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
        if state.shutdown.is_cancelled() {
            return;
        }
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
                // Promote queued files before scheduling background preparations.
                // Process residency also tracks the cached tier for demotion;
                // both paths share one preparation and resource queue.
                state.registry.warm_reconcile(tag);
                state.cache.ensure_cached(tag, CachePriority::Immediate);
            }
            Residency::Disk => {
                // Demote: engine out (idempotent no-op when not loaded),
                // resources on disk. Individual prefetch files queue behind
                // resources needed by process-tier models.
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                if state.shutdown.is_cancelled() {
                    return;
                }
                state.cache.ensure_cached(tag, CachePriority::Prefetch);
            }
            Residency::None => {
                // Retirement only drops logical residency. Content-addressed
                // resources remain available for a future deployment.
                state.dispatcher.remove(tag);
                state.registry.unload(tag).await;
                state.cache.forget(tag);
                state.keys.remove(tag);
                state.devices.remove(tag);
                state.policies.remove(tag);
            }
        }
    }
    // An unload may have waited across shutdown; never undo the final drain.
    if state.shutdown.is_cancelled() {
        return;
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
/// Warming promotes the tag's resource queue priority synchronously, before
/// disk-tier preparations are scheduled later in this reconcile pass.
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

    #[tokio::test]
    async fn shutdown_cannot_be_undrained_by_a_pending_goal_response() {
        let _serial = crate::serving::lease::RUNTIME_TEST_LOCK.lock().unwrap();
        let state = Arc::new(AppState::new(None, crate::state::ServerMode::Standalone).unwrap());
        state.set_draining(true);
        state.shutdown.cancel();
        apply(&state, HeartbeatResponse { drain: false, ..Default::default() }).await;
        assert!(state.is_draining());
        assert!(state.registry.snapshot().is_empty());
        state.lease.shutdown().unwrap();
    }

}
