/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("muna-disk-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn redeployment_only_reserves_missing_files_and_has_no_percentage_reserve() {
    let root = root("redeploy");
    std::fs::write(root.join("weights"), [0; 100]).unwrap();
    let budget = DiskBudget::new(|_| Some(24));
    let manifest = HashMap::from([(root.join("weights"), 100), (root.join("new-code"), 24)]);
    let reservation = budget.try_reserve(&root, &manifest).unwrap().unwrap();
    assert_eq!(reservation.paths, vec![root.join("new-code")]);
    drop(reservation);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn concurrent_preparations_share_reservations_and_cannot_overspend() {
    let root = root("shared");
    let budget = DiskBudget::new(|_| Some(100));
    let shared = HashMap::from([(root.join("shared"), 80)]);
    let a = budget.try_reserve(&root, &shared).unwrap().unwrap();
    let b = budget.try_reserve(&root, &shared).unwrap().unwrap();
    let extra = HashMap::from([(root.join("other"), 21)]);
    assert!(budget.try_reserve(&root, &extra).unwrap().is_none());
    drop(a);
    assert!(budget.try_reserve(&root, &extra).unwrap().is_none());
    drop(b);
    assert!(budget.try_reserve(&root, &extra).unwrap().is_some());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn written_bytes_are_not_counted_twice_and_retired_active_downloads_stay_reserved() {
    let root = root("written");
    let free = Arc::new(AtomicU64::new(100));
    let observed = free.clone();
    let budget = DiskBudget::new(move |_| Some(observed.load(Ordering::SeqCst)));
    let path = root.join("active");
    let a = budget
        .try_reserve(&root, &HashMap::from([(path.clone(), 80)]))
        .unwrap()
        .unwrap();
    budget.started(&path);
    budget.advance(&path, 50);
    free.store(50, Ordering::SeqCst);
    let extra = HashMap::from([(root.join("other"), 20)]);
    assert!(budget.try_reserve(&root, &extra).unwrap().is_some());
    drop(a);
    assert!(budget
        .try_reserve(&root, &HashMap::from([(root.join("too-large"), 21)]))
        .unwrap()
        .is_none());
    budget.finished(&path, true);
    assert!(budget
        .try_reserve(&root, &HashMap::from([(root.join("now-fits"), 21)]))
        .unwrap()
        .is_some());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn releasing_a_reservation_wakes_waiting_preparations() {
    let root = root("wake");
    let budget = DiskBudget::new(|_| Some(10));
    let first = budget
        .try_reserve(&root, &HashMap::from([(root.join("first"), 10)]))
        .unwrap()
        .unwrap();
    let other = budget.clone();
    let path = root.clone();
    let (waiting, mut ready) = tokio::sync::mpsc::unbounded_channel();
    let task = tokio::spawn(async move {
        other
            .reserve(
                &path,
                &HashMap::from([(path.join("second"), 10)]),
                |blocked| {
                    if blocked {
                        let _ = waiting.send(());
                    }
                },
            )
            .await
            .unwrap()
    });
    ready.recv().await.unwrap();
    drop(first);
    let reservation = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    drop(reservation);
    std::fs::remove_dir_all(root).unwrap();
}
