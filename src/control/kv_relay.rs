/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! KV event relay: engine XPUB -> this relay -> the control plane.
//!
//! Runs only in control-plane mode; one relay task per `Ready` model. The task
//! discovers the engine's ZMQ endpoint by predicting the model's `kv`
//! sidecar (the same trivial CPU predictor the preload-claim resolver uses),
//! subscribes to the XPUB, tracks `(epoch, seq)` continuity, batches events
//! over the configured flush window (`--kv-flush-interval`, default 1s --
//! deliberately shorter than the heartbeat, since this window bounds
//! edge-index staleness), and POSTs to the well-known
//! `{control_plane}/v1/kv/events` route -- derived from the control-plane
//! URL the node already has, exactly like the heartbeat route. One
//! consumer, one continuity stream, one `need_snapshot` handler. Fail-soft
//! throughout: POST failures are logged and dropped, a plane without a KV
//! indexer (persistent 404/410) backs the relay off, and the engine-side
//! stream is never back-pressured.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Backoff after the plane rejects the ingest route (404/410: no KV
/// indexer behind the URL). Events in the window are dropped; continuity
/// repairs via snapshot when the route appears.
const NOT_FOUND_BACKOFF: Duration = Duration::from_secs(60);

/// Supervisor: keeps one relay task alive per Ready model.
pub(crate) async fn run(state: Arc<AppState>) {
    let node = state.node.as_ref().expect("kv relay requires node context");
    let ingest_url = format!(
        "{}/v1/kv/events",
        node.control_plane_url.trim_end_matches('/')
    );
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
                ingest_url.clone(),
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
    // The sidecar shares the base model's engine, so predict through the
    // model's own (keyed) Muna instance from the registry.
    let muna = state.registry.ready(tag)?.muna.clone();
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
    /// The plane asked for (or has never received) a snapshot; deltas are
    /// withheld until one flows.
    needs_snapshot: bool,
    /// The ingest route 404/410'd; POSTs are suspended until this passes.
    backoff_until: Option<Instant>,
}

async fn relay_model(
    state: Arc<AppState>,
    tag: String,
    endpoint: String,
    ingest_url: String,
    cancel: CancellationToken
) {
    let node_id = state.node.as_ref()
        .map(|n| n.node_id.clone())
        .unwrap_or_default();
    let token = std::env::var("MUNA_SERVER_TOKEN").ok();
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
    // A fresh session starts snapshot-pending: the plane must not apply
    // deltas before a full baseline.
    let mut edge = EdgeState { needs_snapshot: true, backoff_until: None };
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
                        &ingest_url,
                        token.as_deref(),
                        &node_id,
                        &tag,
                        continuity.epoch.as_deref(),
                        &mut pending,
                        &mut edge
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

/// POST the pending batch to the plane's ingest route. Returns whether a
/// SUB reconnect is needed (the plane wants a snapshot the buffer doesn't
/// hold -- a rejoin forces one from the XPUB).
async fn flush_pending(
    http: &reqwest::Client,
    ingest_url: &str,
    token: Option<&str>,
    node_id: &str,
    tag: &str,
    epoch: Option<&str>,
    pending: &mut Pending,
    edge: &mut EdgeState
) -> bool {
    if let Some(until) = edge.backoff_until {
        if Instant::now() < until {
            // Plane has no KV indexer behind the route: drop the window's
            // events (state repairs by snapshot once the route appears)
            // and stay quiet.
            *pending = Pending::default();
            edge.needs_snapshot = true;
            return false;
        }
        edge.backoff_until = None;
    }
    let Some(epoch) = epoch else {
        return false;
    };
    let Some(seq_range) = pending.seq_range else {
        // Nothing buffered; still reconnect if the plane awaits a snapshot.
        return edge.needs_snapshot;
    };
    if edge.needs_snapshot && !pending.snapshot {
        return true;
    }
    let batch = RelayBatch {
        worker_id: node_id,
        model: tag,
        epoch,
        seq_range,
        snapshot: pending.snapshot,
        events: &pending.events,
    };
    let mut want_reconnect = false;
    let mut request = http.post(ingest_url).json(&batch);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    match request.send().await {
        Ok(response) if response.status().is_success() => {
            let reply: EdgeResponse = response.json().await.unwrap_or_default();
            edge.needs_snapshot = reply.need_snapshot;
            want_reconnect = reply.need_snapshot;
        }
        Ok(response) if matches!(response.status().as_u16(), 404 | 410) => {
            tracing::info!(
                url = %ingest_url,
                status = %response.status(),
                "control plane has no KV ingest route; relay backing off"
            );
            edge.backoff_until = Some(Instant::now() + NOT_FOUND_BACKOFF);
            edge.needs_snapshot = true;
        }
        Ok(response) => {
            tracing::warn!(url = %ingest_url, status = %response.status(), "kv ingest rejected");
        }
        Err(e) => {
            tracing::warn!(url = %ingest_url, error = %e, "kv ingest failed");
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
