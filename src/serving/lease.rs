/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Process-wide GPU device-lease supervisor.
//!
//! nanosgl engines co-resident in this process arbitrate GPU time among
//! themselves through a shared-memory "device lease": a stride-scheduled 
//! fair lock every engine acquires around one step's device work. 
//! The mechanism lives entirely in the engines; this module is the 
//! optional policy side.
//! 
//! muna-server attaches to the same segments to write per-model scheduling
//! weights and the global mode, and to read per-model contention counters.
//! It is never on the acquire path: every write is a single atomic store,
//! every read a handful of atomic loads.
//!
//! The segment appears at `/nsgl-lease-<pid>-dev<N>` when the FIRST engine
//! registers (the Function C runtime dlopens predictors into this process,
//! so `<pid>` is our own pid), and unlinks when the last one deregisters --
//! attachment is therefore lazy and re-probed, and a supervisor finding no
//! segment simply has nothing to supervise yet.
//!
//! Layout safety: `LeaseSegment` here is a field-for-field mirror of the
//! C++ struct (ABI v1); the `layout` test pins every offset the C++
//! `static_assert`s pin. An `abi_version` mismatch (engines built against a
//! newer layout) rejects the segment rather than misinterpreting it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crate::serving::batch::BatchPlan;
use crate::serving::registry::ModelState;
use crate::state::AppState;

/// `LeaseSegment::MAGIC` ('NSGL'): the creator's init-complete handshake.
const MAGIC: u32 = 0x4E53_474C;
/// Segment layout version this mirror understands.
const ABI_VERSION: u32 = 1;
/// Slots per segment (`LeaseSegment::MAX_REGISTRANTS`).
const MAX_REGISTRANTS: usize = 8;
/// Device ordinals probed for segments (per-GPU keying; comfortably above
/// any single-node GPU count we deploy).
const MAX_DEVICES: i32 = 16;
/// Cadence of the weight-reconcile loop in [`run`].
const RECONCILE_INTERVAL: Duration = Duration::from_secs(5);

/// Global arbitration mode, mirror of nanosgl `LeaseMode`. Written by the
/// supervisor, read by every engine at each acquire.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum LeaseMode {
    /// Stride-fair queueing (the default): temporal co-location.
    Fair = 0,
    /// Spatial co-location: engines keep registering and reporting stats
    /// but never block on each other.
    FreeRun = 1,
}

/// One engine's registration + policy + stats slot; byte-for-byte mirror
/// of `LeaseSegment::Registrant` (see the C++ header for field semantics).
#[repr(C)]
struct Registrant {
    /// Owning process id; 0 marks the slot free.
    pid: AtomicI32,
    /// 1 while the registrant is queued for a turn.
    waiting: AtomicU32,
    /// Scheduling weight -- the field this supervisor writes. A share
    /// under contention, never a reservation; 0 = per-model free-run.
    weight: AtomicU32,
    pad0: u32,
    /// Stride virtual time (scaled ns); engine-owned.
    pass_ns: AtomicU64,
    /// Turns completed since registration.
    slots_held: AtomicU64,
    /// Total wall time spent holding turns, in ns.
    ns_held: AtomicU64,
    /// Total wall time spent waiting for turns, in ns.
    ns_waited: AtomicU64,
    /// Engine's steady-clock timestamp of its most recent acquire.
    last_heartbeat_ns: AtomicU64,
    /// NUL-terminated label -- the predictor tag (py2cpp splices
    /// `FXN_PREDICTOR_TAG` as the engine config's `lease_label`).
    label: [u8; 128],
    pad1: u64,
}

/// Byte-for-byte mirror of nanosgl `LeaseSegment` (ABI v1). The pthread
/// mutex/condvar live in fixed 64-byte storage blobs the supervisor never
/// touches -- policy writes and stats reads are lock-free atomics only.
#[repr(C)]
struct LeaseSegment {
    /// 0 while the creator initializes; [`MAGIC`] (store-release) after.
    magic: AtomicU32,
    /// Layout version stamped by the creator.
    abi_version: u32,
    /// Global temporal/spatial dial ([`LeaseMode`] as u32).
    mode: AtomicU32,
    /// Registrant index currently holding the device turn, -1 when free.
    holder: AtomicI32,
    /// Ring virtual time (stride scheduling's global pass).
    virtual_time: AtomicU64,
    pad0: u64,
    /// PROCESS_SHARED (+ robust) pthread mutex; engine-side only.
    mutex_storage: [u8; 64],
    /// PROCESS_SHARED pthread condvar; engine-side only.
    cond_storage: [u8; 64],
    /// Registration slots.
    registrants: [Registrant; MAX_REGISTRANTS],
}

/// Snapshot of one registrant's counters, labeled with its device.
#[derive(Clone)]
pub(crate) struct RegistrantStats {
    /// CUDA device ordinal the segment is keyed on.
    #[allow(dead_code)]
    pub device: i32,
    /// The registrant's label (predictor tag).
    pub label: String,
    /// Current scheduling weight.
    pub weight: u32,
    /// Turns completed since registration.
    #[allow(dead_code)]
    pub slots_held: u64,
    /// Total ns spent holding turns.
    pub ns_held: u64,
    /// Total ns spent waiting for turns (the contention signal).
    pub ns_waited: u64,
}

/// One attached segment mapping. Never unmapped while cached; dropped (and
/// unmapped) only when the ring has fully emptied, so no reference formed
/// inside the supervisor's mutex can dangle.
struct Attachment {
    ptr: *mut LeaseSegment,
}

// SAFETY: the mapping is shared memory accessed exclusively through atomic
// operations (and a read-once label buffer written before the slot's pid is
// published with release ordering).
unsafe impl Send for Attachment {}

impl Attachment {
    fn segment(&self) -> &LeaseSegment {
        // SAFETY: `ptr` is a live, page-aligned MAP_SHARED mapping of at
        // least `size_of::<LeaseSegment>()` bytes, validated (magic +
        // abi_version) at attach time.
        unsafe { &*self.ptr }
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from a successful mmap of exactly this length.
        unsafe {
            libc::munmap(self.ptr.cast(), std::mem::size_of::<LeaseSegment>());
        }
    }
}

/// Attaches lazily to this process's per-device lease segments; the policy
/// (weights, mode) and observability (stats) surface over them. All methods
/// are cheap and non-blocking -- safe to call from async context.
pub(crate) struct LeaseSupervisor {
    /// Cached attachments keyed by device ordinal. A `None` entry marks a
    /// device whose segment was REJECTED (abi skew) so it is not re-probed
    /// (and re-logged) every call.
    segments: Mutex<HashMap<i32, Option<Attachment>>>,
}

impl LeaseSupervisor {

    pub(crate) fn new() -> Self {
        Self { segments: Mutex::new(HashMap::new()) }
    }

    /// Set a registrant's scheduling weight by label (atomic store).
    ///
    /// Weights are shares under contention, never reservations; `0` puts
    /// the model in per-model free-run (spatial), skipping the fair queue.
    /// A label with no live registrant (model not loaded, or its engine
    /// has no lease) is a silent no-op -- the reconcile loop re-asserts
    /// weights periodically, so a late-registering engine converges within
    /// one interval.
    pub(crate) fn set_weight(&self, label: &str, weight: u32) {
        let mut segments = self.segments.lock().unwrap();
        for device in 0..MAX_DEVICES {
            let Some(segment) = attached(&mut segments, device) else {
                continue;
            };
            for slot in &segment.registrants {
                if slot.pid.load(Ordering::Acquire) == 0 {
                    continue;
                }
                if slot_label(slot) == label
                    && slot.weight.load(Ordering::Relaxed) != weight
                {
                    slot.weight.store(weight, Ordering::Relaxed);
                    tracing::info!(
                        label = %label,
                        device = device,
                        weight = weight,
                        "device lease weight set"
                    );
                }
            }
        }
    }

    /// Set the global arbitration mode on every attached segment: `Fair`
    /// (stride-fair queueing, default) or `FreeRun` (spatial; engines keep
    /// registering and reporting stats but never block).
    #[allow(dead_code)]
    pub(crate) fn set_mode(&self, mode: LeaseMode) {
        let mut segments = self.segments.lock().unwrap();
        for device in 0..MAX_DEVICES {
            let Some(segment) = attached(&mut segments, device) else {
                continue;
            };
            segment.mode.store(mode as u32, Ordering::Relaxed);
        }
    }

    /// Snapshot per-registrant counters (`ns_held`, `ns_waited`,
    /// `slots_held`, weight) via atomic loads; never blocks engines.
    pub(crate) fn stats(&self) -> Vec<RegistrantStats> {
        let mut stats = Vec::new();
        let mut segments = self.segments.lock().unwrap();
        for device in 0..MAX_DEVICES {
            let Some(segment) = attached(&mut segments, device) else {
                continue;
            };
            for slot in &segment.registrants {
                if slot.pid.load(Ordering::Acquire) == 0 {
                    continue;
                }
                stats.push(RegistrantStats {
                    device,
                    label: slot_label(slot).to_string(),
                    weight: slot.weight.load(Ordering::Relaxed),
                    slots_held: slot.slots_held.load(Ordering::Relaxed),
                    ns_held: slot.ns_held.load(Ordering::Relaxed),
                    ns_waited: slot.ns_waited.load(Ordering::Relaxed),
                });
            }
        }
        stats
    }
}

/// The cached attachment for `device`, (re-)probing as needed. Returns the
/// segment reference borrowed for the caller's lock scope.
///
/// Staleness: engines unlink the segment when the last one deregisters, so
/// a cached mapping whose slots are all free is dropped and the name is
/// probed fresh -- a NEW ring for the same device (model reload) is a new
/// shm object.
fn attached<'a>(
    segments: &'a mut HashMap<i32, Option<Attachment>>,
    device: i32,
) -> Option<&'a LeaseSegment> {
    if let Some(entry) = segments.get(&device) {
        match entry {
            // Rejected (abi skew): do not re-probe or re-log.
            None => return None,
            Some(attachment) => {
                let live = attachment
                    .segment()
                    .registrants
                    .iter()
                    .any(|slot| slot.pid.load(Ordering::Acquire) != 0);
                if !live {
                    segments.remove(&device);
                } else {
                    // NOTE: reborrow through the map to satisfy the borrow
                    // checker; the attachment is still in place.
                    return segments
                        .get(&device)
                        .and_then(|entry| entry.as_ref())
                        .map(|attachment| attachment.segment());
                }
            }
        }
    }
    match attach(device) {
        AttachOutcome::Attached(attachment) => {
            tracing::info!(device = device, "device lease segment attached");
            segments.insert(device, Some(attachment));
            segments
                .get(&device)
                .and_then(|entry| entry.as_ref())
                .map(|attachment| attachment.segment())
        }
        AttachOutcome::Rejected => {
            segments.insert(device, None);
            None
        }
        AttachOutcome::Absent => None,
    }
}

enum AttachOutcome {
    /// Segment mapped and validated.
    Attached(Attachment),
    /// Segment exists but is unusable (abi skew / bad magic); do not retry.
    Rejected,
    /// No segment for this device (nothing registered yet); retry later.
    Absent,
}

/// Map `/nsgl-lease-<pid>-dev<device>` read-write and validate the header.
fn attach(device: i32) -> AttachOutcome {
    // SAFETY: plain libc calls; the name is NUL-terminated by CString.
    unsafe {
        let pid = libc::getpid();
        let name = std::ffi::CString::new(format!("/nsgl-lease-{pid}-dev{device}"))
            .expect("shm name contains no NUL");
        let fd = libc::shm_open(name.as_ptr(), libc::O_RDWR, 0o600);
        if fd < 0 {
            return AttachOutcome::Absent;
        }
        let map = libc::mmap(
            std::ptr::null_mut(),
            std::mem::size_of::<LeaseSegment>(),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        libc::close(fd);
        if map == libc::MAP_FAILED {
            tracing::warn!(device = device, "device lease segment mmap failed");
            return AttachOutcome::Absent;
        }
        let attachment = Attachment { ptr: map.cast() };
        let segment = attachment.segment();
        if segment.magic.load(Ordering::Acquire) != MAGIC {
            // Creator still initializing (sub-millisecond window) or
            // garbage; treat as absent and let the next probe decide.
            return AttachOutcome::Absent;
        }
        if segment.abi_version != ABI_VERSION {
            tracing::warn!(
                device = device,
                segment_abi = segment.abi_version,
                own_abi = ABI_VERSION,
                "device lease segment ABI mismatch; supervision disabled for this device"
            );
            return AttachOutcome::Rejected;
        }
        AttachOutcome::Attached(attachment)
    }
}

/// The registrant's NUL-terminated label as a str (lossy on invalid UTF-8).
fn slot_label(slot: &Registrant) -> std::borrow::Cow<'_, str> {
    let len = slot
        .label
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(slot.label.len());
    String::from_utf8_lossy(&slot.label[..len])
}

/// Periodic weight-reconcile loop, spawned unconditionally in `main` (the
/// heartbeat is control-plane-gated; lease policy must not be -- co-resident
/// engines arbitrate in standalone mode too).
///
/// Policy: LLM-style models (continuous batching) get weight 3; batch-style
/// models (`Sequential` / `Buffered` dispatch) get weight 1, so an LLM keeps
/// ~75% of contended GPU time against a co-resident batch workload. Weights
/// only bind under contention (work-conserving), so a lone model is
/// unaffected. Idempotent atomic stores make the periodic re-assert free;
/// it also converges engines that registered after the model turned Ready.
pub(crate) async fn run(state: Arc<AppState>) {
    let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        interval.tick().await;
        for (tag, model_state) in state.registry.snapshot() {
            let ModelState::Ready(model) = model_state else {
                continue;
            };
            let weight = match model.plan {
                BatchPlan::Continuous => 3,
                BatchPlan::Sequential | BatchPlan::Buffered { .. } => 1,
            };
            state.lease.set_weight(&tag, weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;

    /// Pins the Rust mirror to nanosgl's `LeaseSegment` ABI v1 -- the same
    /// offsets the C++ `static_assert`s pin. A failure here means the C++
    /// layout changed: bump `ABI_VERSION` on BOTH sides and update this
    /// mirror.
    #[test]
    fn segment_layout_matches_nanosgl_abi_v1() {
        assert_eq!(offset_of!(LeaseSegment, magic), 0);
        assert_eq!(offset_of!(LeaseSegment, abi_version), 4);
        assert_eq!(offset_of!(LeaseSegment, mode), 8);
        assert_eq!(offset_of!(LeaseSegment, holder), 12);
        assert_eq!(offset_of!(LeaseSegment, virtual_time), 16);
        assert_eq!(offset_of!(LeaseSegment, mutex_storage), 32);
        assert_eq!(offset_of!(LeaseSegment, cond_storage), 96);
        assert_eq!(offset_of!(LeaseSegment, registrants), 160);
        assert_eq!(offset_of!(Registrant, pid), 0);
        assert_eq!(offset_of!(Registrant, waiting), 4);
        assert_eq!(offset_of!(Registrant, weight), 8);
        assert_eq!(offset_of!(Registrant, pass_ns), 16);
        assert_eq!(offset_of!(Registrant, slots_held), 24);
        assert_eq!(offset_of!(Registrant, ns_held), 32);
        assert_eq!(offset_of!(Registrant, ns_waited), 40);
        assert_eq!(offset_of!(Registrant, last_heartbeat_ns), 48);
        assert_eq!(offset_of!(Registrant, label), 56);
        assert_eq!(size_of::<Registrant>(), 192);
        assert_eq!(size_of::<LeaseSegment>(), 160 + 8 * 192);
    }

    /// Attaching with no engine registered anywhere is a clean no-op.
    #[test]
    fn absent_segments_are_a_noop() {
        let supervisor = LeaseSupervisor::new();
        assert!(supervisor.stats().is_empty());
        supervisor.set_weight("@test/model", 3);
        supervisor.set_mode(LeaseMode::FreeRun);
    }
}
