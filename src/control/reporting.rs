/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Interruptible status transport. Dropping a request cannot retract a report
//! already received by the plane, so both directions carry snapshot versions.

use reqwest::RequestBuilder;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::protocol::{HeartbeatResponse, HeartbeatVersion};

pub(super) enum Report {
    Changed,
    Stopped,
    Complete(Option<HeartbeatResponse>),
}

pub(super) async fn exchange(
    request: RequestBuilder,
    version: HeartbeatVersion,
    changed: &Notify,
    shutdown: &CancellationToken
) -> Report {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => Report::Stopped,
        _ = changed.notified() => Report::Changed,
        response = send(request, version) => Report::Complete(response),
    }
}

pub(super) async fn send(
    request: RequestBuilder,
    version: HeartbeatVersion
) -> Option<HeartbeatResponse> {
    let response = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(%error, "heartbeat failed");
            return None;
        }
    };
    if !response.status().is_success() {
        tracing::warn!(status = %response.status(), "heartbeat rejected");
        return None;
    }
    let response: HeartbeatResponse = match response.json().await {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(%error, "malformed heartbeat response");
            return None;
        }
    };
    if response.heartbeat != version {
        tracing::warn!(expected = ?version, received = ?response.heartbeat,
            "heartbeat response version mismatch; control plane must support sequenced heartbeats");
        return None;
    }
    Some(response)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::{extract::State, routing::post, Json, Router};
    use serde_json::{json, Value};
    use tokio::sync::mpsc;

    use super::*;

    struct Stub {
        received: mpsc::UnboundedSender<Value>,
        release_first: Notify,
    }

    async fn heartbeat(State(stub): State<Arc<Stub>>, Json(body): Json<Value>) -> Json<Value> {
        stub.received.send(body.clone()).unwrap();
        if body["sequence"] == 1 {
            stub.release_first.notified().await;
        }
        Json(json!({"session": body["session"], "sequence": body["sequence"], "drain": false}))
    }

    async fn start() -> (String, Arc<Stub>, mpsc::UnboundedReceiver<Value>, tokio::task::JoinHandle<()>) {
        let (received, reports) = mpsc::unbounded_channel();
        let stub = Arc::new(Stub { received, release_first: Notify::new() });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = Router::new().route("/", post(heartbeat)).with_state(stub.clone());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap(); });
        (url, stub, reports, server)
    }

    #[tokio::test]
    async fn status_change_reports_before_the_previous_request_finishes() {
        let (url, stub, mut reports, server) = start().await;
        let client = reqwest::Client::new();
        let changed = Arc::new(Notify::new());
        let shutdown = CancellationToken::new();
        let first = HeartbeatVersion { session: 42, sequence: 1 };
        let request = client.post(&url).json(&first);
        let notifying = changed.clone();
        let stop = shutdown.clone();
        let pending = tokio::spawn(async move { exchange(request, first, &notifying, &stop).await });
        let body = tokio::time::timeout(Duration::from_secs(2), reports.recv()).await.unwrap().unwrap();
        assert_eq!(body["sequence"], 1);
        changed.notify_one();
        assert!(matches!(tokio::time::timeout(Duration::from_secs(2), pending).await.unwrap().unwrap(), Report::Changed));

        // The plane is deliberately still holding sequence 1. A fresh report
        // must get through without waiting for that handler to return.
        let second = HeartbeatVersion { sequence: 2, ..first };
        let response = tokio::time::timeout(Duration::from_secs(2),
            exchange(client.post(&url).json(&second), second, &changed, &shutdown)
        ).await.unwrap();
        assert!(matches!(response, Report::Complete(Some(reply)) if reply.heartbeat == second));
        assert_eq!(reports.recv().await.unwrap()["sequence"], 2);
        stub.release_first.notify_one();
        server.abort();
    }

    #[tokio::test]
    async fn mismatched_and_unversioned_replies_cannot_apply_goals() {
        let (url, _stub, _reports, server) = start().await;
        let client = reqwest::Client::new();
        let expected = HeartbeatVersion { session: 42, sequence: 9 };
        for reported in [
            HeartbeatVersion { sequence: 8, ..expected },
            HeartbeatVersion { session: 41, ..expected },
            HeartbeatVersion::default(),
        ] {
            let reply = send(client.post(&url).json(&reported), expected).await;
            assert!(reply.is_none());
        }
        server.abort();
    }

    #[tokio::test]
    async fn notification_bursts_coalesce_and_shutdown_interrupts_the_wait() {
        let (url, _stub, mut reports, server) = start().await;
        let client = reqwest::Client::new();
        let changed = Notify::new();
        let shutdown = CancellationToken::new();
        let version = HeartbeatVersion { session: 42, sequence: 2 };
        for _ in 0..100 { changed.notify_one(); }
        let report = exchange(client.post(&url).json(&version), version, &changed, &shutdown).await;
        assert!(matches!(report, Report::Changed));
        let report = exchange(client.post(&url).json(&version), version, &changed, &shutdown).await;
        assert!(matches!(report, Report::Complete(Some(_))));
        assert_eq!(reports.recv().await.unwrap()["sequence"], 2);
        assert!(reports.try_recv().is_err());
        shutdown.cancel();
        let report = exchange(client.post(&url).json(&version), version, &changed, &shutdown).await;
        assert!(matches!(report, Report::Stopped));
        server.abort();
    }
}
