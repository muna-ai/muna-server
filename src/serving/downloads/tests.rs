/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::*;
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

fn path(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "muna-queue-{}-{}-{name}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn enqueue(
    downloads: &Downloads,
    path: &Path,
    priority: Priority,
    events: &mpsc::UnboundedSender<PathBuf>,
    gate: Option<oneshot::Receiver<()>>,
) -> JoinHandle<Result<(), String>> {
    let downloads = downloads.clone();
    let path = path.to_owned();
    let events = events.clone();
    tokio::spawn(async move {
        let started_path = path.clone();
        downloads
            .download(&path, priority, None, move |_| {
                Box::pin(async move {
                    events.send(started_path).unwrap();
                    if let Some(gate) = gate {
                        gate.await.unwrap();
                    }
                    Ok(())
                })
            })
            .await
    })
}

async fn registered(downloads: &Downloads, path: &Path, subscribers: usize) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let count = downloads
                .0
                .queue
                .lock()
                .unwrap()
                .jobs
                .get(path)
                .map(|job| job.subscribers.lock().unwrap().len());
            if count == Some(subscribers) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("download did not register");
}

async fn next(events: &mut mpsc::UnboundedReceiver<PathBuf>) -> PathBuf {
    tokio::time::timeout(Duration::from_secs(3), events.recv())
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn process_files_overtake_prefetch_with_fifo_and_no_preemption() {
    let downloads = Downloads::default();
    let (events, mut received) = mpsc::unbounded_channel();
    let (release, gate) = oneshot::channel();
    let active = path("active");
    let first = enqueue(
        &downloads,
        &active,
        Priority::default(),
        &events,
        Some(gate),
    );
    assert_eq!(next(&mut received).await, active);
    let background = path("background");
    let second = enqueue(&downloads, &background, Priority::default(), &events, None);
    registered(&downloads, &background, 1).await;
    let process = Priority::default();
    process.set_process(true);
    let urgent_a = path("urgent-a");
    let third = enqueue(&downloads, &urgent_a, process.clone(), &events, None);
    registered(&downloads, &urgent_a, 1).await;
    let urgent_b = path("urgent-b");
    let fourth = enqueue(&downloads, &urgent_b, process, &events, None);
    registered(&downloads, &urgent_b, 1).await;
    assert!(received.try_recv().is_err());
    release.send(()).unwrap();
    assert_eq!(next(&mut received).await, urgent_a);
    assert_eq!(next(&mut received).await, urgent_b);
    assert_eq!(next(&mut received).await, background);
    for job in [first, second, third, fourth] {
        job.await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn queued_prefetch_can_be_promoted_without_resubmission() {
    let downloads = Downloads::default();
    let (events, mut received) = mpsc::unbounded_channel();
    let (release, gate) = oneshot::channel();
    let active = enqueue(
        &downloads,
        &path("active"),
        Priority::default(),
        &events,
        Some(gate),
    );
    next(&mut received).await;
    let background_path = path("background");
    let background = enqueue(
        &downloads,
        &background_path,
        Priority::default(),
        &events,
        None,
    );
    registered(&downloads, &background_path, 1).await;
    let priority = Priority::default();
    let promoted_path = path("promoted");
    let promoted = enqueue(&downloads, &promoted_path, priority.clone(), &events, None);
    registered(&downloads, &promoted_path, 1).await;
    priority.set_process(true);
    release.send(()).unwrap();
    assert_eq!(next(&mut received).await, promoted_path);
    assert_eq!(next(&mut received).await, background_path);
    for job in [active, background, promoted] {
        job.await.unwrap().unwrap();
    }
}

#[tokio::test]
async fn canceling_one_waiter_preserves_a_shared_active_transfer_and_progress() {
    let downloads = Downloads::default();
    let destination = path("shared");
    let (events, mut received) = mpsc::unbounded_channel();
    let (release, gate) = oneshot::channel();
    let first = {
        let downloads = downloads.clone();
        let path = destination.clone();
        tokio::spawn(async move {
            downloads
                .download(&path, Priority::default(), None, move |progress| {
                    Box::pin(async move {
                        events.send(PathBuf::from("started")).unwrap();
                        gate.await.unwrap();
                        progress(32, Some(32));
                        Ok(())
                    })
                })
                .await
        })
    };
    next(&mut received).await;
    let bytes = Arc::new(AtomicU64::new(0));
    let progress_bytes = bytes.clone();
    let progress: DownloadProgressFn = Arc::new(move |increment, _| {
        progress_bytes.fetch_add(increment, Ordering::Relaxed);
    });
    let second = {
        let downloads = downloads.clone();
        let path = destination.clone();
        tokio::spawn(async move {
            downloads
                .download(&path, Priority::default(), Some(progress), |_| {
                    Box::pin(async {
                        panic!("duplicate physical transfer");
                    })
                })
                .await
        })
    };
    registered(&downloads, &destination, 2).await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    registered(&downloads, &destination, 1).await;
    release.send(()).unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(bytes.load(Ordering::Relaxed), 32);
}

#[tokio::test]
async fn canceled_queued_jobs_are_skipped_and_cache_is_rechecked_at_dequeue() {
    let downloads = Downloads::default();
    let (events, mut received) = mpsc::unbounded_channel();
    let (release, gate) = oneshot::channel();
    let active = enqueue(
        &downloads,
        &path("active"),
        Priority::default(),
        &events,
        Some(gate),
    );
    next(&mut received).await;
    let canceled_path = path("canceled");
    let canceled = enqueue(
        &downloads,
        &canceled_path,
        Priority::default(),
        &events,
        None,
    );
    registered(&downloads, &canceled_path, 1).await;
    canceled.abort();
    assert!(canceled.await.unwrap_err().is_cancelled());
    let cached_path = path("cached");
    let cached = enqueue(&downloads, &cached_path, Priority::default(), &events, None);
    registered(&downloads, &cached_path, 1).await;
    tokio::fs::write(&cached_path, b"already cached")
        .await
        .unwrap();
    release.send(()).unwrap();
    active.await.unwrap().unwrap();
    cached.await.unwrap().unwrap();
    assert!(received.try_recv().is_err());
    assert!(downloads.0.queue.lock().unwrap().jobs.is_empty());
    tokio::fs::remove_file(cached_path).await.unwrap();
}

#[tokio::test]
async fn failures_reach_all_waiters_release_capacity_and_allow_retry() {
    let downloads = Downloads::default();
    let destination = path("failure");
    let (release, gate) = oneshot::channel();
    let first = {
        let downloads = downloads.clone();
        let path = destination.clone();
        tokio::spawn(async move {
            downloads
                .download(&path, Priority::default(), None, |_| {
                    Box::pin(async move {
                        gate.await.unwrap();
                        Err("failed transfer".into())
                    })
                })
                .await
        })
    };
    registered(&downloads, &destination, 1).await;
    let second = {
        let downloads = downloads.clone();
        let path = destination.clone();
        tokio::spawn(async move {
            downloads
                .download(&path, Priority::default(), None, |_| {
                    Box::pin(async { panic!("duplicate") })
                })
                .await
        })
    };
    registered(&downloads, &destination, 2).await;
    release.send(()).unwrap();
    assert_eq!(first.await.unwrap().unwrap_err(), "failed transfer");
    assert_eq!(second.await.unwrap().unwrap_err(), "failed transfer");
    downloads
        .download(&destination, Priority::default(), None, |_| {
            Box::pin(async { Ok(()) })
        })
        .await
        .unwrap();
    assert!(downloads
        .download(&destination, Priority::default(), None, |_| Box::pin(
            async { panic!("transfer panic") }
        ))
        .await
        .is_err());
    downloads
        .download(&destination, Priority::default(), None, |_| {
            Box::pin(async { Ok(()) })
        })
        .await
        .unwrap();
}
