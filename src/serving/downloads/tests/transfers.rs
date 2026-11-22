/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::*;

#[tokio::test]
async fn malformed_ranges_fail_without_publication_or_temporary_files() {
    // Wrong start, ignored range, short/long bodies, changed total, missing
    // Content-Range and HTTP failures must never publish a preallocated file.
    for mode in [3, 4, 5, 6, 7, 8, 9] {
        let resource = Resource::new(17);
        resource.mode.store(mode, Ordering::Relaxed);
        let fixture = Fixture::new(&[("resource", resource.clone())]).await;
        let downloads = downloads();
        assert!(
            fixture
                .enqueue(&downloads, "resource", Priority::default(), None, None)
                .await
                .unwrap()
                .is_err(),
            "mode {mode}"
        );
        assert!(!fixture.path("resource").exists());
        fixture.assert_clean();
        assert!(downloads.0.queue.lock().unwrap().jobs.is_empty());
        resource.mode.store(0, Ordering::Relaxed);
        complete(fixture.enqueue(&downloads, "resource", Priority::default(), None, None)).await;
        assert_eq!(
            std::fs::read(fixture.path("resource")).unwrap(),
            resource.bytes
        );
        fixture.assert_clean();
    }
}

#[tokio::test]
async fn failures_reach_shared_waiters_and_do_not_block_the_next_job() {
    let resource = Resource::gated(17);
    resource.mode.store(9, Ordering::Relaxed);
    let gate = resource.gate.clone().unwrap();
    let fixture = Fixture::new(&[("bad", resource), ("good", Resource::new(5))]).await;
    let downloads = downloads();
    let first = fixture.enqueue(&downloads, "bad", Priority::default(), None, None);
    until(|| fixture.bodies("bad").len() == 2).await;
    let second = fixture.enqueue(&downloads, "bad", Priority::default(), None, None);
    registered(&downloads, &fixture.path("bad"), 2).await;
    let good = fixture.enqueue(&downloads, "good", Priority::default(), None, None);
    registered(&downloads, &fixture.path("good"), 1).await;
    gate.add_permits(100);
    let a = first.await.unwrap().unwrap_err();
    assert_eq!(second.await.unwrap().unwrap_err(), a);
    complete(good).await;
    assert!(!fixture.path("bad").exists());
    fixture.assert_clean();
}

#[tokio::test]
async fn streams_resume_the_same_response_across_quanta() {
    for mode in [1, 2] {
        let resource = Resource::new(23);
        resource.mode.store(mode, Ordering::Relaxed);
        let fixture = Fixture::new(&[("stream", resource.clone())]).await;
        let path = fixture.path("stream");
        let mut transfer = Transfer::new(fixture.http.clone(), fixture.url("stream"), &path);
        let bytes = Arc::new(AtomicU64::new(0));
        let observed = bytes.clone();
        let progress: DownloadProgressFn = Arc::new(move |increment, _| {
            observed.fetch_add(increment, Ordering::Relaxed);
        });
        let limits = Limits {
            range: 4,
            quantum: 8,
        };
        assert!(!transfer.advance(limits, progress.clone()).await.unwrap());
        assert!(!path.exists());
        assert_eq!(bytes.load(Ordering::Relaxed), 8);
        assert!(!transfer.advance(limits, progress.clone()).await.unwrap());
        assert_eq!(bytes.load(Ordering::Relaxed), 16);
        assert!(transfer.advance(limits, progress).await.unwrap());
        transfer.publish().await.unwrap();
        assert_eq!(std::fs::read(path).unwrap(), resource.bytes);
        assert_eq!(bytes.load(Ordering::Relaxed), 23);
        assert_eq!(
            fixture.origin.requests.lock().unwrap().len(),
            1,
            "stream restarted after yielding"
        );
        fixture.assert_clean();
    }
}

#[tokio::test]
async fn empty_resources_are_published_without_range_underflow() {
    let fixture = Fixture::new(&[("empty", Resource::new(0))]).await;
    complete(fixture.enqueue(&downloads(), "empty", Priority::default(), None, None)).await;
    assert_eq!(std::fs::metadata(fixture.path("empty")).unwrap().len(), 0);
    fixture.assert_clean();
}

#[tokio::test]
async fn a_panicking_subscriber_does_not_strand_other_resources() {
    let fixture = Fixture::new(&[("panic", Resource::new(5)), ("next", Resource::new(5))]).await;
    let downloads = downloads();
    let progress: DownloadProgressFn = Arc::new(|_, _| panic!("subscriber failed"));
    assert!(fixture
        .enqueue(
            &downloads,
            "panic",
            Priority::default(),
            Some(progress),
            None
        )
        .await
        .unwrap()
        .is_err());
    assert!(!fixture.path("panic").exists());
    complete(fixture.enqueue(&downloads, "next", Priority::default(), None, None)).await;
    fixture.assert_clean();
}

#[tokio::test]
async fn probe_size_must_match_the_admitted_manifest() {
    let fixture = Fixture::new(&[("resource", Resource::new(17))]).await;
    let mut transfer = Transfer::new(
        fixture.http.clone(),
        fixture.url("resource"),
        &fixture.path("resource"),
    )
    .with_size(Some(18));
    let progress: DownloadProgressFn = Arc::new(|_, _| {});
    assert!(transfer
        .advance(
            Limits {
                range: 4,
                quantum: 8
            },
            progress
        )
        .await
        .unwrap_err()
        .contains("manifest"));
    transfer.discard().await;
    assert!(!fixture.path("resource").exists());
    fixture.assert_clean();
}
