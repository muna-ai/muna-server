/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::sync::atomic::AtomicUsize;

use super::*;

fn registry(delay: Duration, fail: bool) -> (ModelRegistry, Arc<AtomicUsize>) {
    let loads = Arc::new(AtomicUsize::new(0));
    let counting = loads.clone();
    let loader: Loader = Arc::new(move |_, _| {
        let loads = counting.clone();
        Box::pin(async move {
            loads.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(delay).await;
            if fail { return Err("load failed".into()); }
            Ok(Loaded {
                signature: Signature { inputs: vec![], outputs: vec![] },
                muna: Arc::new(Muna::new(None, None)),
                download: Duration::ZERO,
                load: delay,
                build: None,
            })
        })
    });
    let registry = ModelRegistry::with_loader(loader, Arc::default(), None,
        Arc::new(NotificationCenter::default()), Arc::new(crate::platform::NullPlatform));
    (registry, loads)
}

#[tokio::test]
async fn retirement_fences_admission_and_waits_for_active_execution() {
    let (registry, loads) = registry(Duration::ZERO, false);
    registry.inner.keys.insert("@test/model".into(), "old-key".into());
    let old = registry.ensure_ready("@test/model").await.unwrap();
    let execution = old.admit().await.unwrap();
    registry.retire("@test/model");
    assert!(registry.ready("@test/model").is_none());
    assert!(old.admit().await.is_err());
    let unloading = tokio::spawn({
        let registry = registry.clone();
        async move { registry.unload("@test/model").await }
    });
    tokio::task::yield_now().await;
    assert!(!unloading.is_finished());
    // Even if goals change while the engine is draining, observations remain
    // attached to the old actual deployment until native execution has ended.
    registry.inner.keys.insert("@test/model".into(), "new-key".into());
    let states = registry.snapshot();
    let ModelState::Ready(actual) = &states[0].1 else { panic!("old engine must remain accounted for"); };
    assert_eq!(actual.key_hash, Some(crate::client::key_hash("old-key")));
    drop(execution);
    unloading.await.unwrap();
    assert!(registry.snapshot().is_empty());
    let new = registry.ensure_ready("@test/model").await.unwrap();
    assert_eq!(new.key_hash, Some(crate::client::key_hash("new-key")));
    assert!(!Arc::ptr_eq(&old, &new));
    assert_eq!(loads.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn superseded_initialization_is_deleted_before_another_load_starts() {
    let (registry, loads) = registry(Duration::from_millis(100), false);
    registry.inner.keys.insert("@test/model".into(), "old-key".into());
    registry.warm("@test/model");
    registry.retire("@test/model");
    let unloading = tokio::spawn({
        let registry = registry.clone();
        async move { registry.unload("@test/model").await }
    });
    tokio::task::yield_now().await;
    registry.inner.keys.insert("@test/model".into(), "new-key".into());
    registry.warm("@test/model");
    assert!(registry.ready_tags().is_empty());
    let states = registry.snapshot();
    let ModelState::Loading { key_hash, .. } = &states[0].1 else { panic!("initialization must remain tracked"); };
    assert_eq!(key_hash, &Some(crate::client::key_hash("old-key")));
    unloading.await.unwrap();
    assert!(registry.snapshot().is_empty());
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    let new = registry.ensure_ready("@test/model").await.unwrap();
    assert_eq!(new.key_hash, Some(crate::client::key_hash("new-key")));
    assert_eq!(loads.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_failed_attempt_retains_its_original_deployment() {
    let (registry, _) = registry(Duration::ZERO, true);
    registry.inner.keys.insert("@test/model".into(), "old-key".into());
    registry.warm("@test/model");
    let mut completion = registry.inner.models.get("@test/model").unwrap().watch.subscribe();
    completion.changed().await.unwrap();
    registry.inner.keys.insert("@test/model".into(), "new-key".into());
    let states = registry.snapshot();
    let ModelState::Failed { key_hash, .. } = &states[0].1 else { panic!("failure must remain tracked"); };
    assert_eq!(key_hash, &Some(crate::client::key_hash("old-key")));
    registry.unload("@test/model").await;
    assert!(registry.snapshot().is_empty());
}
