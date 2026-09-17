/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Process-tier downloads run ahead of disk-tier prefetch.
//!
//! Every download on a node shares one disk writeback budget. On a network
//! root volume that budget is ~130 MB/s while the CDN delivers >1 GB/s, so
//! an ungated `process` load that shares the disk with 80 GB of `disk`-tier
//! prefetch is ready when the whole prefetch is: ~10 min instead of the
//! ~30 s its own bytes need. The gate counts in-flight priority
//! (process-tier) downloads; prefetch waits until the count is zero.
//!
//! The count is bumped synchronously by whoever spawns a priority load, so
//! a prefetch spawned later in the same reconcile pass cannot observe zero
//! before the load's task has started.

use tokio::sync::watch;

/// Counts in-flight priority downloads; see the module docs.
pub(crate) struct DownloadGate {
    inflight: watch::Sender<usize>,
}

/// Held for the duration of a priority download; dropping it releases one
/// slot. The holder decides where the download ends (the registry loader
/// drops it once resources are on disk, before the engine load).
pub(crate) struct PriorityGuard {
    inflight: watch::Sender<usize>,
}

impl DownloadGate {

    pub(crate) fn new() -> Self {
        Self { inflight: watch::Sender::new(0) }
    }

    /// Register a priority download. Synchronous on purpose: callers bump
    /// the count before spawning so a prefetch spawned afterwards in the
    /// same pass queues behind it.
    pub(crate) fn enter_priority(&self) -> PriorityGuard {
        self.inflight.send_modify(|n| *n += 1);
        PriorityGuard { inflight: self.inflight.clone() }
    }

    /// Resolve once no priority download is in flight.
    pub(crate) async fn wait_idle(&self) {
        let mut rx = self.inflight.subscribe();
        // `wait_for` checks the current value first, so an idle gate
        // returns at once. The channel cannot close while `self` (a
        // sender) lives, so the error arm is unreachable in practice.
        let _ = rx.wait_for(|n| *n == 0).await;
    }

    /// Number of priority downloads in flight.
    #[cfg(test)]
    pub(crate) fn inflight(&self) -> usize {
        *self.inflight.borrow()
    }
}

impl Drop for PriorityGuard {
    fn drop(&mut self) {
        self.inflight.send_modify(|n| *n = n.saturating_sub(1));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn idle_gate_resolves_immediately() {
        let gate = DownloadGate::new();
        assert_eq!(gate.inflight(), 0);
        tokio::time::timeout(Duration::from_millis(50), gate.wait_idle())
            .await
            .expect("idle gate must not block");
    }

    #[tokio::test]
    async fn guard_blocks_until_dropped() {
        let gate = Arc::new(DownloadGate::new());
        let guard = gate.enter_priority();
        assert_eq!(gate.inflight(), 1);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), gate.wait_idle())
                .await
                .is_err(),
            "gate must block while a guard is held"
        );
        let waiter = {
            let gate = gate.clone();
            tokio::spawn(async move { gate.wait_idle().await })
        };
        drop(guard);
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("waiter must resolve once the guard drops")
            .unwrap();
        assert_eq!(gate.inflight(), 0);
    }

    #[tokio::test]
    async fn every_guard_must_drop() {
        let gate = DownloadGate::new();
        let a = gate.enter_priority();
        let b = gate.enter_priority();
        assert_eq!(gate.inflight(), 2);
        drop(a);
        assert_eq!(gate.inflight(), 1);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), gate.wait_idle())
                .await
                .is_err()
        );
        drop(b);
        tokio::time::timeout(Duration::from_millis(50), gate.wait_idle())
            .await
            .expect("gate idle after the last guard drops");
    }
}
