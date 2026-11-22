/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::*;

#[tokio::test]
async fn urgent_download_interrupts_a_file_and_resumes_without_repeating_ranges() {
    let background = Resource::gated(32);
    let gate = background.gate.clone().unwrap();
    let fixture = Fixture::new(&[
        ("background", background.clone()),
        ("urgent", Resource::new(5)),
    ])
    .await;
    let downloads = downloads();
    let bytes = Arc::new(AtomicU64::new(0));
    let observed = bytes.clone();
    let progress: DownloadProgressFn = Arc::new(move |increment, _| {
        observed.fetch_add(increment, Ordering::Relaxed);
    });
    let first = fixture.enqueue(
        &downloads,
        "background",
        Priority::default(),
        Some(progress),
        None,
    );
    until(|| fixture.bodies("background").len() == 2).await;
    let urgent = Priority::default();
    urgent.set_process(true);
    let second = fixture.enqueue(&downloads, "urgent", urgent, None, None);
    registered(&downloads, &fixture.path("urgent"), 1).await;
    gate.add_permits(2);
    complete(second).await;
    assert!(!first.is_finished());
    assert!(
        !fixture.path("background").exists(),
        "partial file was published"
    );
    assert_eq!(bytes.load(Ordering::Relaxed), 8);
    gate.add_permits(100);
    complete(first).await;
    assert_eq!(
        std::fs::read(fixture.path("background")).unwrap(),
        background.bytes
    );
    assert_eq!(bytes.load(Ordering::Relaxed), 32);
    let mut ranges: Vec<_> = fixture
        .bodies("background")
        .iter()
        .map(|r| (r.start, r.end))
        .collect();
    ranges.sort_unstable();
    assert_eq!(
        ranges,
        (0..32)
            .step_by(4)
            .map(|start| (start, start + 3))
            .collect::<Vec<_>>()
    );
    fixture.assert_clean();
}

#[tokio::test]
async fn cache_hits_bypass_a_blocked_transfer() {
    let resource = Resource::gated(16);
    let gate = resource.gate.clone().unwrap();
    let fixture = Fixture::new(&[("blocked", resource)]).await;
    let downloads = downloads();
    let active = fixture.enqueue(&downloads, "blocked", Priority::default(), None, None);
    until(|| fixture.bodies("blocked").len() == 2).await;
    let cached = fixture.path("cached");
    std::fs::write(&cached, b"already cached").unwrap();
    tokio::time::timeout(
        Duration::from_secs(1),
        downloads.download(
            &fixture.http,
            "http://invalid.invalid",
            &cached,
            Priority::default(),
            None,
            None,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!active.is_finished());
    gate.add_permits(100);
    complete(active).await;
}

#[tokio::test]
async fn inference_then_deployment_then_background() {
    let blocker = Resource::gated(16);
    let gate = blocker.gate.clone().unwrap();
    let fixture = Fixture::new(&[
        ("blocker", blocker),
        ("background", Resource::new(5)),
        ("deployment", Resource::new(5)),
        ("inference", Resource::new(5)),
    ])
    .await;
    let downloads = downloads();
    let active = fixture.enqueue(&downloads, "blocker", Priority::default(), None, None);
    until(|| fixture.bodies("blocker").len() == 2).await;
    let background = fixture.enqueue(&downloads, "background", Priority::default(), None, None);
    registered(&downloads, &fixture.path("background"), 1).await;
    let deployment = Priority::default();
    deployment.set_deployment(true);
    let deploying = fixture.enqueue(&downloads, "deployment", deployment, None, None);
    registered(&downloads, &fixture.path("deployment"), 1).await;
    let inference = Priority::default();
    let predicting = fixture.enqueue(&downloads, "inference", inference.clone(), None, None);
    registered(&downloads, &fixture.path("inference"), 1).await;
    // Promote an already queued job without resubmitting it.
    inference.set_process(true);
    gate.add_permits(100);
    for task in [active, background, deploying, predicting] {
        complete(task).await;
    }
    let requests = fixture.origin.requests.lock().unwrap();
    let first = |name| {
        requests
            .iter()
            .position(|request| request.name == name)
            .unwrap()
    };
    assert!(first("inference") < first("deployment"));
    assert!(first("deployment") < first("background"));
}

#[tokio::test]
async fn a_small_model_finishes_its_manifest_before_large_models_next_shard() {
    let blocker = Resource::gated(32);
    let gate = blocker.gate.clone().unwrap();
    let fixture = Fixture::new(&[
        ("blocker", blocker),
        ("large-weights", Resource::new(40)),
        ("large-binary", Resource::new(4)),
        ("small-weights", Resource::new(4)),
        ("small-binary", Resource::new(4)),
    ])
    .await;
    let downloads = downloads();
    let active = fixture.enqueue(&downloads, "blocker", Priority::default(), None, None);
    until(|| fixture.bodies("blocker").len() == 2).await;
    let (large_priority, large_tracker) = fixture
        .model("large", &["large-weights", "large-binary"])
        .await;
    let (small_priority, small_tracker) = fixture
        .model("small", &["small-weights", "small-binary"])
        .await;
    let large = fixture.enqueue_model(
        &downloads,
        &["large-weights", "large-binary"],
        large_priority,
        large_tracker,
    );
    registered(&downloads, &fixture.path("large-weights"), 1).await;
    let small = fixture.enqueue_model(
        &downloads,
        &["small-weights", "small-binary"],
        small_priority,
        small_tracker,
    );
    registered(&downloads, &fixture.path("small-weights"), 1).await;
    gate.add_permits(100);
    complete(small).await;
    complete(large).await;
    complete(active).await;
    let requests = fixture.origin.requests.lock().unwrap();
    let first = |name| {
        requests
            .iter()
            .position(|request| request.name == name)
            .unwrap()
    };
    assert!(first("small-weights") < first("large-weights"));
    assert!(first("small-binary") < first("large-binary"));
}

#[tokio::test]
async fn a_shared_subscriber_inherits_progress_and_survives_creator_cancellation() {
    let resource = Resource::gated(32);
    let gate = resource.gate.clone().unwrap();
    let fixture = Fixture::new(&[("shared", resource)]).await;
    let downloads = downloads();
    let first = fixture.enqueue(&downloads, "shared", Priority::default(), None, None);
    until(|| fixture.bodies("shared").len() == 2).await;
    gate.add_permits(1);
    until(|| {
        downloads.0.queue.lock().unwrap().jobs[&fixture.path("shared")]
            .state
            .lock()
            .unwrap()
            .available
            == 4
    })
    .await;
    let bytes = Arc::new(AtomicU64::new(0));
    let observed = bytes.clone();
    let progress: DownloadProgressFn = Arc::new(move |increment, _| {
        observed.fetch_add(increment, Ordering::Relaxed);
    });
    let priority = Priority::default();
    priority.set_process(true);
    let second = fixture.enqueue(&downloads, "shared", priority, Some(progress), None);
    registered(&downloads, &fixture.path("shared"), 2).await;
    until(|| bytes.load(Ordering::Relaxed) == 4).await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    registered(&downloads, &fixture.path("shared"), 1).await;
    gate.add_permits(100);
    complete(second).await;
    assert_eq!(bytes.load(Ordering::Relaxed), 32);
    assert_eq!(fixture.bodies("shared").len(), 8);
    fixture.assert_clean();
}

#[tokio::test]
async fn canceled_unstarted_jobs_are_skipped_and_publication_is_rechecked() {
    let resource = Resource::gated(16);
    let gate = resource.gate.clone().unwrap();
    let fixture = Fixture::new(&[
        ("blocked", resource),
        ("canceled", Resource::new(4)),
        ("cached", Resource::new(4)),
    ])
    .await;
    let downloads = downloads();
    let active = fixture.enqueue(&downloads, "blocked", Priority::default(), None, None);
    until(|| fixture.bodies("blocked").len() == 2).await;
    let canceled = fixture.enqueue(&downloads, "canceled", Priority::default(), None, None);
    registered(&downloads, &fixture.path("canceled"), 1).await;
    canceled.abort();
    assert!(canceled.await.unwrap_err().is_cancelled());
    let cached = fixture.enqueue(&downloads, "cached", Priority::default(), None, None);
    registered(&downloads, &fixture.path("cached"), 1).await;
    std::fs::write(fixture.path("cached"), b"done").unwrap();
    gate.add_permits(100);
    complete(active).await;
    complete(cached).await;
    assert!(fixture
        .origin
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|request| request.name == "blocked"));
    assert!(downloads.0.queue.lock().unwrap().jobs.is_empty());
}

#[tokio::test]
async fn aging_gives_large_models_one_turn_without_returning_to_file_fifo() {
    let blocker = Resource::gated(16);
    let gate = blocker.gate.clone().unwrap();
    let fixture = Fixture::new(&[
        ("blocker", blocker),
        ("aged", Resource::new(40)),
        ("small", Resource::new(4)),
    ])
    .await;
    let downloads = downloads();
    let active = fixture.enqueue(&downloads, "blocker", Priority::default(), None, None);
    until(|| fixture.bodies("blocker").len() == 2).await;
    let (priority, tracker) = fixture.model("aged-model", &["aged"]).await;
    let aged = fixture.enqueue(&downloads, "aged", priority, None, Some(tracker));
    registered(&downloads, &fixture.path("aged"), 1).await;
    let (priority, tracker) = fixture.model("small-model", &["small"]).await;
    let small = fixture.enqueue(&downloads, "small", priority, None, Some(tracker));
    registered(&downloads, &fixture.path("small"), 1).await;
    downloads.0.queue.lock().unwrap().jobs[&fixture.path("aged")]
        .state
        .lock()
        .unwrap()
        .last_served = Instant::now() - MAX_WAIT - Duration::from_secs(1);
    gate.add_permits(100);
    for task in [small, aged, active] {
        complete(task).await;
    }
    let requests = fixture.origin.requests.lock().unwrap();
    let first = |name| {
        requests
            .iter()
            .position(|request| request.name == name)
            .unwrap()
    };
    let resumed = requests
        .iter()
        .position(|request| request.name == "aged" && request.start == 8)
        .unwrap();
    assert!(first("aged") < first("small"));
    assert!(
        first("small") < resumed,
        "aging should grant one quantum, not the whole file"
    );
}
