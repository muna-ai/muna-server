/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! KV event relay: engine XPUB -> this relay -> edge indexers.
//!
//! Runs only in control-plane mode; one relay task per `Ready` model. The task
//! discovers the engine's ZMQ endpoint by predicting the model's `kv`
//! sidecar (the same trivial CPU predictor the preload-claim resolver uses),
//! subscribes to the XPUB, tracks `(epoch, seq)` continuity, batches events
//! over the configured flush window (`--kv-flush-interval`, default 1s --
//! deliberately shorter than the heartbeat, since this window bounds
//! edge-index staleness), and POSTs to every controller-provided edge
//! callback URL. Fail-soft throughout: callback failures are logged and
//! dropped; the engine-side stream is never back-pressured.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use muna::types::{Acceleration, Value};
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;
use zeromq::{Socket, SocketRecv};

use crate::control::protocol::{EdgeResponse, RelayBatch};
use crate::serving::predict;
use crate::state::AppState;

/// How often the model scan looks for newly Ready / unloaded models.
const SCAN_INTERVAL: Duration = Duration::from_secs(5);

/// Reconnect the SUB if the engine goes silent this long (a rejoin forces a
/// snapshot, repairing any missed events).
const RECV_TIMEOUT: Duration = Duration::from_secs(60);

/// Backoff after a failed connect / subscribe.
const RETRY_DELAY: Duration = Duration::from_secs(5);

/// Supervisor: keeps one relay task alive per Ready model.
pub(crate) async fn run(state: Arc<AppState>) {
    let node = state.node.as_ref().expect("kv relay requires node context");
    let callbacks = node.event_callbacks.subscribe();
    let mut tasks: HashMap<String, (CancellationToken, tokio::task::JoinHandle<()>)> = HashMap::new();
    // Models whose kv-sidecar discovery failed (not KV-routed); avoid
    // re-predicting the sidecar on every scan.
    let mut not_kv_routed: HashSet<String> = HashSet::new();
    let mut scan = tokio::time::interval(SCAN_INTERVAL);
    scan.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        scan.tick().await;
        let ready: HashSet<String> = state.registry.ready_tags().into_iter().collect();
        tasks.retain(|tag, (cancel, handle)| {
            if !ready.contains(tag) {
                cancel.cancel();
                return false;
            }
            !handle.is_finished()
        });
        not_kv_routed.retain(|tag| ready.contains(tag));
        for tag in &ready {
            if tasks.contains_key(tag) || not_kv_routed.contains(tag) {
                continue;
            }
            let Some(endpoint) = discover_endpoint(&state, tag).await else {
                not_kv_routed.insert(tag.clone());
                continue;
            };
            tracing::info!(tag = %tag, endpoint = %endpoint, "kv relay attached");
            let cancel = CancellationToken::new();
            let handle = tokio::spawn(relay_model(
                state.clone(),
                tag.clone(),
                endpoint,
                callbacks.clone(),
                cancel.clone()
            ));
            tasks.insert(tag.clone(), (cancel, handle));
        }
    }
}

/// Predict the model's `kv` sidecar locally; its first string output is the
/// engine's ZMQ endpoint. A prediction failure means the model has no kv
/// sidecar, i.e. it is not KV-routed.
async fn discover_endpoint(state: &Arc<AppState>, tag: &str) -> Option<String> {
    let muna = state.muna.clone();
    let sidecar = format!("{tag}:kv");
    let result = predict::run(move || async move {
        let inputs = HashMap::from([("_".to_string(), Value::Null)]);
        muna.predictions.create(
            &sidecar,
            Some(inputs),
            Some(Acceleration::LocalAuto),
            None,
            None
        ).await
    }).await;
    let prediction = match result {
        Ok(prediction) => prediction,
        Err(e) => {
            tracing::debug!(tag = %tag, error = %e, "no kv sidecar; model is not KV-routed");
            return None;
        }
    };
    if let Some(error) = prediction.error {
        tracing::debug!(tag = %tag, error = %error, "kv sidecar prediction failed");
        return None;
    }
    prediction.results?.into_iter().find_map(|value| match value {
        Value::String(endpoint) => Some(endpoint),
        _ => None,
    })
}

/// Buffered events awaiting the next flush.
#[derive(Default)]
struct Pending {
    events: Vec<JsonValue>,
    seq_range: Option<(u64, u64)>,
    snapshot: bool,
}

/// Stream-continuity tracker: state is only trusted while the stream is
/// provably continuous -- epoch constant, seq contiguous.
#[derive(Default)]
struct Continuity {
    epoch: Option<String>,
    last_seq: Option<u64>,
}

#[derive(Debug, PartialEq)]
enum Admit {
    /// Snapshot batch: reset buffered state, then apply.
    Snapshot,
    /// Contiguous delta: append.
    Delta,
    /// Discontinuity (epoch change or seq gap): reconnect for a snapshot.
    Gap,
}

impl Continuity {
    fn admit(&mut self, epoch: &str, seq: u64, snapshot: bool) -> Admit {
        if snapshot {
            // A snapshot re-baselines the stream unconditionally.
            self.epoch = Some(epoch.to_string());
            self.last_seq = Some(seq);
            return Admit::Snapshot;
        }
        let continuous =
            self.epoch.as_deref().is_none_or(|e| e == epoch) &&
            self.last_seq.is_none_or(|last| seq == last + 1);
        if !continuous {
            return Admit::Gap;
        }
        self.epoch = Some(epoch.to_string());
        self.last_seq = Some(seq);
        Admit::Delta
    }
}

struct EdgeState {
    /// Edge asked for (or has never received) a snapshot; deltas are
    /// withheld until one flows.
    needs_snapshot: bool,
}

async fn relay_model(
    state: Arc<AppState>,
    tag: String,
    endpoint: String,
    callbacks: tokio::sync::watch::Receiver<Vec<String>>,
    cancel: CancellationToken
) {
    let node_id = state.node.as_ref()
        .map(|n| n.node_id.clone())
        .unwrap_or_default();
    let flush_interval = state.node.as_ref()
        .map(|n| n.kv_flush_interval)
        .unwrap_or(Duration::from_secs(1));
    let http = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build() {
        Ok(client) => client,
        Err(e) => {
            tracing::error!(error = %e, "failed to build kv relay client");
            return;
        }
    };
    let mut edges: HashMap<String, EdgeState> = HashMap::new();
    'session: loop {
        if cancel.is_cancelled() {
            return;
        }
        let mut sub = zeromq::SubSocket::new();
        if let Err(e) = sub.connect(&endpoint).await {
            tracing::warn!(tag = %tag, error = %e, "kv relay connect failed");
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(RETRY_DELAY) => continue 'session,
            }
        }
        if let Err(e) = sub.subscribe("").await {
            tracing::warn!(tag = %tag, error = %e, "kv relay subscribe failed");
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(RETRY_DELAY) => continue 'session,
            }
        }
        let mut continuity = Continuity::default();
        let mut pending = Pending::default();
        let mut flush = tokio::time::interval(flush_interval);
        flush.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = flush.tick() => {
                    let want_reconnect = flush_pending(
                        &http,
                        &node_id,
                        &tag,
                        continuity.epoch.as_deref(),
                        &mut pending,
                        &mut edges,
                        &callbacks
                    ).await;
                    if want_reconnect {
                        // A rejoin forces a snapshot from the XPUB.
                        continue 'session;
                    }
                }
                result = tokio::time::timeout(RECV_TIMEOUT, sub.recv()) => {
                    let message = match result {
                        // Engine silent past the expected cadence: rejoin so
                        // any missed events are repaired by a snapshot.
                        Err(_) => continue 'session,
                        Ok(Err(e)) => {
                            tracing::warn!(tag = %tag, error = %e, "kv relay recv failed");
                            continue 'session;
                        }
                        Ok(Ok(message)) => message,
                    };
                    let Some((seq, payload)) = parse_message(message) else {
                        continue;
                    };
                    let msg_epoch = payload
                        .get("epoch")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let snapshot = payload
                        .get("snapshot")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let events: Vec<JsonValue> = payload
                        .get("events")
                        .and_then(|v| v.as_array())
                        .cloned()
                        .unwrap_or_default();
                    match continuity.admit(&msg_epoch, seq, snapshot) {
                        Admit::Snapshot => {
                            // A snapshot supersedes any buffered deltas
                            // (reset-then-set); events after it stay in order.
                            pending = Pending {
                                events,
                                seq_range: Some((seq, seq)),
                                snapshot: true,
                            };
                        }
                        Admit::Delta => {
                            pending.events.extend(events);
                            pending.seq_range = Some(match pending.seq_range {
                                Some((lo, _)) => (lo, seq),
                                None => (seq, seq),
                            });
                        }
                        Admit::Gap => {
                            tracing::warn!(
                                tag = %tag,
                                seq,
                                "kv event discontinuity; reconnecting for snapshot"
                            );
                            continue 'session;
                        }
                    }
                }
            }
        }
    }
}

/// Decode one XPUB message: `("kv", seq_be64, json payload)`.
fn parse_message(message: zeromq::ZmqMessage) -> Option<(u64, JsonValue)> {
    let frames = message.into_vec();
    if frames.len() != 3 || frames[0].as_ref() != b"kv" {
        return None;
    }
    let seq_bytes: [u8; 8] = frames[1].as_ref().try_into().ok()?;
    let seq = u64::from_be_bytes(seq_bytes);
    let payload: JsonValue = serde_json::from_slice(frames[2].as_ref()).ok()?;
    Some((seq, payload))
}

/// POST the pending batch to every edge callback. Returns whether a SUB
/// reconnect is needed (an edge wants a snapshot the buffer doesn't hold).
///
/// A snapshot batch goes to every edge, not just the one that requested it:
/// the snapshot consumes a publisher seq, so withholding it from current
/// edges would open a gap in their stream. Snapshots are reset-then-set and
/// therefore idempotent for edges that were already current.
async fn flush_pending(
    http: &reqwest::Client,
    node_id: &str,
    tag: &str,
    epoch: Option<&str>,
    pending: &mut Pending,
    edges: &mut HashMap<String, EdgeState>,
    callbacks: &tokio::sync::watch::Receiver<Vec<String>>
) -> bool {
    let urls = callbacks.borrow().clone();
    edges.retain(|url, _| urls.iter().any(|u| u == url));
    for url in &urls {
        // A new edge starts snapshot-pending: it must not apply deltas
        // before a full baseline.
        edges.entry(url.clone()).or_insert(EdgeState { needs_snapshot: true });
    }
    let Some(epoch) = epoch else {
        return false;
    };
    let Some(seq_range) = pending.seq_range else {
        // Nothing buffered; still reconnect if any edge awaits a snapshot.
        return edges.values().any(|e| e.needs_snapshot);
    };
    let batch = RelayBatch {
        worker_id: node_id,
        model: tag,
        epoch,
        seq_range,
        snapshot: pending.snapshot,
        events: &pending.events,
    };
    let mut want_reconnect = false;
    for url in &urls {
        let edge = edges.get_mut(url).expect("edge state ensured above");
        if edge.needs_snapshot && !pending.snapshot {
            want_reconnect = true;
            continue;
        }
        match http.post(url).json(&batch).send().await {
            Ok(response) if response.status().is_success() => {
                let reply: EdgeResponse = response.json().await.unwrap_or_default();
                if reply.need_snapshot {
                    edge.needs_snapshot = true;
                    want_reconnect = true;
                } else {
                    edge.needs_snapshot = false;
                }
            }
            Ok(response) => {
                tracing::warn!(url = %url, status = %response.status(), "kv callback rejected");
            }
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "kv callback failed");
            }
        }
    }
    *pending = Pending::default();
    want_reconnect
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use serde_json::json;
    use zeromq::SocketSend;

    use super::*;

    fn frames(topic: &str, seq: u64, payload: &JsonValue) -> zeromq::ZmqMessage {
        zeromq::ZmqMessage::try_from(vec![
            Bytes::copy_from_slice(topic.as_bytes()),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from(serde_json::to_vec(payload).unwrap()),
        ]).unwrap()
    }

    #[test]
    fn parse_message_decodes_wire_frames() {
        let payload = json!({
            "epoch": "abc",
            "seq": 42,
            "snapshot": false,
            "events": [{ "type": "stored", "block_hashes": ["ff"], "tier": "device" }]
        });
        let (seq, parsed) = parse_message(frames("kv", 42, &payload)).unwrap();
        assert_eq!(seq, 42);
        assert_eq!(parsed, payload);
    }

    #[test]
    fn parse_message_rejects_wrong_topic_or_arity() {
        let payload = json!({ "seq": 1 });
        assert!(parse_message(frames("nope", 1, &payload)).is_none());
        let two_frames = zeromq::ZmqMessage::try_from(vec![
            Bytes::from_static(b"kv"),
            Bytes::from_static(b"x"),
        ]).unwrap();
        assert!(parse_message(two_frames).is_none());
    }

    #[test]
    fn continuity_contiguous_deltas_admit() {
        let mut c = Continuity::default();
        assert_eq!(c.admit("e1", 1, false), Admit::Delta);
        assert_eq!(c.admit("e1", 2, false), Admit::Delta);
        assert_eq!(c.admit("e1", 3, false), Admit::Delta);
    }

    #[test]
    fn continuity_seq_gap_is_discontinuous() {
        let mut c = Continuity::default();
        assert_eq!(c.admit("e1", 1, false), Admit::Delta);
        assert_eq!(c.admit("e1", 3, false), Admit::Gap);
    }

    #[test]
    fn continuity_epoch_change_is_discontinuous() {
        let mut c = Continuity::default();
        assert_eq!(c.admit("e1", 1, false), Admit::Delta);
        assert_eq!(c.admit("e2", 2, false), Admit::Gap);
    }

    #[test]
    fn continuity_snapshot_rebaselines() {
        let mut c = Continuity::default();
        assert_eq!(c.admit("e1", 1, false), Admit::Delta);
        // Engine restarted: new epoch arrives as a snapshot (rejoin path).
        assert_eq!(c.admit("e2", 17, true), Admit::Snapshot);
        assert_eq!(c.admit("e2", 18, false), Admit::Delta);
    }

    /// Mock publisher (same crate) driving framing over a real socket:
    /// snapshot first (as the XPUB does on subscribe), then a delta.
    #[tokio::test]
    async fn pub_sub_round_trip_preserves_framing() {
        let mut publisher = zeromq::PubSocket::new();
        let endpoint = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut sub = zeromq::SubSocket::new();
        sub.connect(&endpoint.to_string()).await.unwrap();
        sub.subscribe("").await.unwrap();
        // PUB drops messages sent before the subscription handshake lands;
        // retry-publish until the subscriber observes the snapshot.
        let snapshot = json!({
            "epoch": "e1",
            "seq": 7,
            "snapshot": true,
            "events": [{ "type": "stored", "block_hashes": ["aa", "bb"], "tier": "device" }]
        });
        let received = loop {
            publisher.send(frames("kv", 7, &snapshot)).await.unwrap();
            match tokio::time::timeout(Duration::from_millis(200), sub.recv()).await {
                Ok(Ok(message)) => break message,
                _ => continue,
            }
        };
        let (seq, payload) = parse_message(received).unwrap();
        let mut continuity = Continuity::default();
        assert_eq!(seq, 7);
        assert_eq!(continuity.admit(
            payload["epoch"].as_str().unwrap(),
            seq,
            payload["snapshot"].as_bool().unwrap()
        ), Admit::Snapshot);
        // A contiguous delta follows the snapshot.
        let delta = json!({
            "epoch": "e1",
            "seq": 8,
            "snapshot": false,
            "events": [{ "type": "removed", "block_hashes": ["aa"], "tier": "device" }]
        });
        publisher.send(frames("kv", 8, &delta)).await.unwrap();
        let message = tokio::time::timeout(Duration::from_secs(5), sub.recv())
            .await
            .unwrap()
            .unwrap();
        let (seq, payload) = parse_message(message).unwrap();
        assert_eq!(seq, 8);
        assert_eq!(continuity.admit(
            payload["epoch"].as_str().unwrap(),
            seq,
            payload["snapshot"].as_bool().unwrap()
        ), Admit::Delta);
    }
}
