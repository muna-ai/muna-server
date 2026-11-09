/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Host-owned Corby runtime and deployment scheduling policy.
//! Predictor stubs discover this runtime; no Rust/C++ shared-memory layout is
//! mirrored here. Native leases and invocation-wide intent use the same engine.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use corby::{Context, Runtime, RuntimeOptions, Tenancy, Work, CAP_COOPERATIVE};

#[derive(Clone)]
pub(crate) struct RegistrantStats {
    pub device: i32,
    pub label: String,
    pub weight: u32,
    #[allow(dead_code)]
    pub slots_held: u64,
    pub ns_held: u64,
    pub ns_waited: u64,
    // Corby's stable stats currently expose grants, not mid-forward yields.
    pub yields: u64,
}

pub(crate) struct LeaseSupervisor {
    context: Context,
    runtime: Mutex<Runtime>,
    // A deployment reservation survives idle unloads until reconciliation
    // confirms retirement. Corby enforces exclusivity among loaded engines.
    reservations: Mutex<HashMap<String, Vec<u32>>>,
}

impl LeaseSupervisor {

    pub(crate) fn new() -> Result<Self, String> {
        let runtime = Runtime::start(RuntimeOptions::default())
            .map_err(|e| format!("failed to start Corby runtime: {e}"))?;
        Ok(Self {
            context: runtime.supervisor(),
            runtime: Mutex::new(runtime),
            reservations: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn shutdown(&self) -> Result<(), String> {
        self.runtime
            .lock()
            .map_err(|_| "Corby runtime lock poisoned".to_owned())?
            .shutdown()
            .map_err(|e| e.to_string())
    }

    pub(crate) fn reserve_dedicated(&self, tag: &str, devices: &[u32]) -> bool {
        let mut reservations = self.reservations.lock().unwrap();
        if reservations
            .iter()
            .any(|(owner, ids)| owner != tag && ids.iter().any(|id| devices.contains(id)))
            || reservations.get(tag).is_some_and(|ids| ids != devices)
        {
            return false;
        }
        reservations.insert(tag.to_owned(), devices.to_vec());
        true
    }

    pub(crate) fn retain_reservations(&self, resident: impl Fn(&str) -> bool) {
        self.reservations
            .lock()
            .unwrap()
            .retain(|tag, _| resident(tag));
    }

    pub(crate) fn protected(&self, devices: &[u32]) -> bool {
        self.reservations
            .lock()
            .unwrap()
            .values()
            .any(|ids| ids.iter().any(|id| devices.contains(id)))
    }

    /// Exactly one registration must cover the entire immutable assignment.
    fn registration(&self, label: &str, devices: &[u32]) -> corby::Result<u64> {
        let mut expected: Vec<i32> = devices
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| corby::Error(corby::Status::InvalidArgument)))
            .collect::<Result<_, _>>()?;
        expected.sort_unstable();
        if expected.is_empty() || expected.windows(2).any(|ids| ids[0] == ids[1]) {
            return Err(corby::Error(corby::Status::InvalidArgument));
        }
        let mut matching = Vec::new();
        for id in self.context.engine_ids()? {
            if self.context.engine_label(id)? == label {
                matching.push(id);
            }
        }
        if matching.len() != 1 || self.context.engine_devices(matching[0])? != expected {
            return Err(corby::Error(corby::Status::NotFound));
        }
        Ok(matching[0])
    }

    pub(crate) fn devices_for(&self, label: &str) -> Option<Vec<u32>> {
        let ids = self.context.engine_ids().ok()?;
        let matching: Vec<_> = ids
            .into_iter()
            .filter(|&id| {
                self.context
                    .engine_label(id)
                    .is_ok_and(|name| name == label)
            })
            .collect();
        if matching.len() != 1 {
            return None;
        }
        self.context
            .engine_devices(matching[0])
            .ok()?
            .into_iter()
            .map(|id| u32::try_from(id).ok())
            .collect()
    }

    pub(crate) fn supports(&self, label: &str, devices: &[u32]) -> bool {
        let Ok(id) = self.registration(label, devices) else {
            return false;
        };
        let Ok(assigned) = self.context.engine_devices(id) else {
            return false;
        };
        let Ok(ids) = self.context.engine_ids() else {
            return false;
        };
        ids.into_iter().all(|id| {
            let Ok(peer_devices) = self.context.engine_devices(id) else {
                return false;
            };
            if !peer_devices.iter().any(|device| assigned.contains(device)) {
                return true;
            }
            self.context
                .engine_stats(id)
                .is_ok_and(|stats| stats.capabilities & CAP_COOPERATIVE != 0)
        })
    }

    /// Called after loading, before the registry publishes Ready. The backend
    /// validates dedicated conflicts and cooperative peers atomically.
    pub(crate) fn configure(
        &self,
        label: &str,
        devices: &[u32],
        dedicated: bool,
    ) -> Result<(), String> {
        let id = self.registration(label, devices)
            .map_err(|e| format!("Predictor must register its assigned device group with Corby; rebuild the predictor: {e}"))?;
        self.context
            .set_engine_tenancy(
                id,
                if dedicated {
                    Tenancy::Dedicated
                } else {
                    Tenancy::Shared
                },
            )
            .map_err(|e| format!("Corby tenancy rejected for {label}: {e}"))
    }

    fn enter(
        &self,
        label: &str,
        devices: &[u32],
        dedicated: bool,
    ) -> Result<BusyGuard, muna::MunaError> {
        if (dedicated || self.protected(devices)) && !self.supports(label, devices) {
            return Err(muna::MunaError::Native(
                "Engine does not support cooperative Corby priority; rebuild the predictor.".into(),
            ));
        }
        let work = self
            .registration(label, devices)
            .and_then(|id| self.context.begin_engine(id))
            .map_err(|e| {
                muna::MunaError::Native(format!("Corby invocation admission failed: {e}"))
            })?;
        Ok(BusyGuard { _work: work })
    }

    pub(crate) fn set_weight_on(&self, device: i32, label: &str, weight: u32) {
        let Ok(ids) = self.context.engine_ids() else {
            return;
        };
        for id in ids {
            if self
                .context
                .engine_label(id)
                .is_ok_and(|name| name == label)
                && self
                    .context
                    .engine_devices(id)
                    .is_ok_and(|devices| devices.contains(&device))
            {
                if let Err(error) = self.context.set_engine_weight(id, weight) {
                    tracing::warn!(%label, %error, "Corby weight update failed");
                }
            }
        }
    }

    /// Expand atomic group counters per device for the existing demand solver
    /// and GPU-second heartbeat aggregation. Each group has one common weight.
    pub(crate) fn stats(&self) -> Vec<RegistrantStats> {
        let mut result = Vec::new();
        let Ok(ids) = self.context.engine_ids() else {
            return result;
        };
        for id in ids {
            let (Ok(stats), Ok(label), Ok(devices)) = (
                self.context.engine_stats(id),
                self.context.engine_label(id),
                self.context.engine_devices(id),
            ) else {
                continue;
            };
            for device in devices {
                result.push(RegistrantStats {
                    device,
                    label: label.clone(),
                    weight: stats.weight,
                    slots_held: stats.grants,
                    ns_held: stats.held_ns,
                    ns_waited: stats.waited_ns,
                    yields: 0,
                });
            }
        }
        result
    }
}

pub(crate) struct ModelLease {
    pub supervisor: Arc<LeaseSupervisor>,
    pub label: String,
    pub devices: Vec<u32>,
    pub dedicated: bool,
}

impl ModelLease {
    pub(crate) fn enter(&self) -> Result<BusyGuard, muna::MunaError> {
        self.supervisor
            .enter(&self.label, &self.devices, self.dedicated)
    }
}

// Owned by the registry execution guard on the native blocking task. Caller
// disconnects must not drop this before native execution and cleanup finish.
pub(crate) struct BusyGuard {
    _work: Work<'static>,
}

#[cfg(test)]
pub(crate) static RUNTIME_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn corby_supervision_preserves_priority_and_native_lifetimes() {
        let _serial = RUNTIME_TEST_LOCK.lock().unwrap();
        let supervisor = Arc::new(LeaseSupervisor::new().unwrap());
        assert!(supervisor.stats().is_empty());
        assert!(supervisor.reserve_dedicated("dedicated", &[0, 1]));
        assert!(!supervisor.reserve_dedicated("other", &[1, 2]));
        assert!(!supervisor.reserve_dedicated("dedicated", &[1, 2]));
        let shared = supervisor
            .context
            .register("shared", &[0, 1], 1, Tenancy::Shared, true)
            .unwrap();
        let dedicated = supervisor
            .context
            .register("dedicated", &[0, 1], 1, Tenancy::Shared, true)
            .unwrap();
        assert!(supervisor.supports("dedicated", &[1, 0]));
        assert!(!supervisor.supports("dedicated", &[0]));
        supervisor.configure("dedicated", &[0, 1], true).unwrap();
        supervisor.set_weight_on(1, "dedicated", 7);
        assert_eq!(dedicated.stats().unwrap().weight, 7);
        let peer = shared.begin().unwrap();
        let held = peer.acquire(None).unwrap();
        let busy = supervisor.enter("dedicated", &[0, 1], true).unwrap();
        assert!(held.should_yield(Duration::from_secs(3600)).unwrap());
        drop(held);
        assert_eq!(
            peer.acquire(Some(Duration::ZERO)).err().unwrap().0,
            corby::Status::Timeout
        );
        drop(busy);
        assert!(peer.acquire(Some(Duration::ZERO)).is_ok());
        assert_eq!(supervisor.stats().len(), 4);

        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let owner = supervisor.clone();
        let caller = tokio::spawn(crate::serving::predict::run(None, move || async move {
            let busy = owner.enter("dedicated", &[0, 1], true)?;
            started_tx.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            drop(busy);
            done_tx.send(()).unwrap();
            Ok::<_, muna::MunaError>(())
        }));
        started_rx.await.unwrap();
        caller.abort();
        assert_eq!(dedicated.stats().unwrap().active_work, 1);
        done_rx.await.unwrap();
        assert_eq!(dedicated.stats().unwrap().active_work, 0);
        drop(peer);
        drop(dedicated);
        drop(shared);
        supervisor.retain_reservations(|_| false);
        assert!(!supervisor.protected(&[0]));
        supervisor.shutdown().unwrap();
    }
}
