/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::*;

fn event() -> ModelLoadEvent {
    ModelLoadEvent {
        id: uuid::Uuid::new_v4().to_string(), tag: "@test/fast".into(),
        created: std::time::SystemTime::now().into(), duration_ms: 0.125, download_ms: 0.0,
        server_version: "test".into(), device_ids: Some(vec![]), build: Some("same-build".into()),
        vram_mb: None, gpu_family: None, gpu_name: None, concurrent_loads: 0,
    }
}

#[test]
fn completion_wire_format_matches_the_control_plane() {
    let wire = serde_json::json!({
        "id": "d540b149-fdce-444b-8542-216838792a97",
        "tag": "@test/fast", "created": "2026-11-26T00:00:00Z",
        "duration_ms": 0.125, "download_ms": 0.0, "server_version": "test",
        "device_ids": [], "build": "build", "vram_mb": null,
        "gpu_family": "cpu", "gpu_name": "CPU", "concurrent_loads": 0
    });
    let event: ModelLoadEvent = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(event).unwrap(), wire);
    let reply: crate::control::protocol::HeartbeatResponse = serde_json::from_value(serde_json::json!({
        "session": 7, "sequence": 8, "drain": false,
        "event_acks": ["d540b149-fdce-444b-8542-216838792a97"]
    })).unwrap();
    assert_eq!(reply.event_acks, vec!["d540b149-fdce-444b-8542-216838792a97"]);
}

struct Cache(PathBuf);

impl Cache {
    fn new() -> Self { Self(std::env::temp_dir().join(format!("muna-load-events-{}", uuid::Uuid::new_v4()))) }
    fn open(&self) -> LoadEvents { LoadEvents::open(&self.0, "node", Arc::default()).unwrap() }
}

impl Drop for Cache {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

#[test]
fn retries_restart_and_partial_acknowledgements_preserve_load_identity() {
    let cache = Cache::new();
    let events = cache.open();
    let first = event();
    let second = event();
    events.record(first.clone());
    events.record(second.clone());
    // An acknowledgement cannot discard a fact that hasn't reached disk.
    events.acknowledge(&[first.id.clone()]);
    assert!(events.batch().is_empty());
    events.flush().unwrap();
    assert_eq!(events.batch().len(), 2);
    drop(events);

    let events = cache.open();
    let replayed = events.batch();
    let replayed_first = replayed.iter().find(|event| event.id == first.id).unwrap();
    assert_eq!(replayed_first.created, first.created);
    assert_eq!(replayed_first.duration_ms, 0.125);
    events.acknowledge(&[first.id.clone(), "../../not-an-event".into()]);
    assert_eq!(events.batch()[0].id, second.id);
    // A crash before acknowledgement cleanup replays, rather than loses, a fact.
    assert_eq!(cache.open().batch().len(), 2);
    events.flush().unwrap();
    let events = cache.open();
    assert_eq!(events.batch().len(), 1);
    assert_eq!(events.batch()[0].id, second.id);
    events.acknowledge(&[first.id]);
    assert_eq!(events.batch().len(), 1);
    events.acknowledge(&[second.id]);
    events.flush().unwrap();
    assert!(cache.open().batch().is_empty());
}

#[test]
fn bounded_batches_do_not_drop_events_and_node_identity_isolated() {
    let cache = Cache::new();
    let events = cache.open();
    for _ in 0..70 { events.record(event()); }
    events.flush().unwrap();
    assert_eq!(events.batch().len(), 64);
    let acks = events.batch().into_iter().map(|event| event.id).collect::<Vec<_>>();
    events.acknowledge(&acks);
    events.flush().unwrap();
    assert_eq!(events.batch().len(), 6);
    assert!(LoadEvents::open(&cache.0, "replacement", Arc::default()).unwrap().batch().is_empty());
}

#[test]
fn failed_disk_write_is_retained_and_incomplete_writes_are_ignored() {
    let cache = Cache::new();
    let events = cache.open();
    let pending = event();
    events.record(pending.clone());
    std::fs::remove_dir(&events.path).unwrap();
    std::fs::write(&events.path, b"unavailable directory").unwrap();
    assert!(events.flush().is_err());
    assert!(events.batch().is_empty());
    std::fs::remove_file(&events.path).unwrap();
    std::fs::create_dir(&events.path).unwrap();
    events.flush().unwrap();
    std::fs::write(events.path.join("interrupted.tmp"), b"partial JSON").unwrap();
    assert_eq!(cache.open().batch()[0].id, pending.id);
    std::fs::write(events.path.join("corrupt.json"), b"broken").unwrap();
    assert!(LoadEvents::open(&cache.0, "node", Arc::default()).is_err());
}

#[tokio::test]
async fn durable_completion_wakes_reporting_without_waiting_for_a_heartbeat_tick() {
    let cache = Cache::new();
    let events = Arc::new(cache.open());
    let shutdown = CancellationToken::new();
    let worker = tokio::spawn({
        let events = events.clone();
        let shutdown = shutdown.clone();
        async move { events.run(&shutdown).await }
    });
    events.record(event());
    tokio::time::timeout(Duration::from_secs(2), async {
        while events.batch().is_empty() { events.notifications.status.notified().await; }
    }).await.unwrap();
    shutdown.cancel();
    worker.await.unwrap();
    assert_eq!(cache.open().batch().len(), 1);
}
