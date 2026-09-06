/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! The `Platform` contract every vendor backend implements, the device
//! snapshot it reports, and process-wide platform detection.

use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use super::family::{GpuFamily, GpuVendor};
use super::host;

/// One device as reported in heartbeats.
#[derive(Serialize)]
pub(crate) struct DeviceMetrics {
    /// Device index (0-based).
    pub index: u32,
    /// Device vendor (nvidia, apple, amd).
    pub vendor: GpuVendor,
    /// Device name (e.g. "NVIDIA H100", "Apple M4 Pro").
    pub name: String,
    /// Standardized GPU family derived from the device name; `None` for
    /// devices outside the fleet vocabulary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<GpuFamily>,
    /// Total VRAM in MB. On Apple Silicon, this is the recommended max
    /// working set size.
    pub vram_total_mb: u64,
    /// Currently allocated VRAM in MB. `None` if the backend cannot report
    /// this.
    pub vram_used_mb: Option<u64>,
    /// Free VRAM in MB. `None` if the backend cannot report this.
    pub vram_free_mb: Option<u64>,
    /// GPU compute utilization as a percentage (0-100). `None` on Apple
    /// Silicon.
    pub utilization_pct: Option<u32>,
    /// GPU temperature in degrees Celsius. `None` on Apple Silicon.
    pub temperature_c: Option<u32>,
}

/// Outcome of a one-time driver warm, for the boot log. The caller times
/// the call, so the report carries only what it cannot observe itself.
pub(crate) struct WarmReport {
    /// Devices whose driver context was brought up.
    pub devices: usize,
}

/// Everything the node reports about the machine it runs on. The vendor
/// methods are abstract; the host methods vary by OS rather than by
/// vendor, so they default to the shared implementations in `host` and
/// are overridden only where a platform genuinely differs (or in tests,
/// to pin values).
pub(crate) trait Platform: Send + Sync {

    /// GPU vendor.
    fn vendor(&self) -> GpuVendor;

    /// Bring the vendor driver up once so the first engine load does not
    /// pay for it: map the runtime library, create the per-device primary
    /// context (or equivalent), and keep both alive for the process
    /// lifetime. Idempotent and fail-soft; called from `spawn_blocking` at
    /// boot. `Ok(None)` means the platform has nothing to warm (or already
    /// did).
    fn warm(&self) -> Result<Option<WarmReport>, String>;

    /// Per-device snapshot for the heartbeat `gpus` array.
    fn devices(&self) -> Vec<DeviceMetrics>;

    /// Device memory in use across all devices, for the load-time VRAM
    /// delta the registry records. `None` if the platform cannot say.
    fn memory_used_mb(&self) -> Option<u64>;

    /// Free and total space in MB of the filesystem containing `path`.
    /// Feeds capacity reporting: resources are never deleted, so the
    /// control plane must see remaining disk to know when to stop placing
    /// models on this node.
    fn disk_space_mb(&self, path: &Path) -> Option<(u64, u64)> {
        host::disk_space_mb(path)
    }

    /// Total physical RAM in MB.
    fn system_ram_mb(&self) -> Option<u64> {
        host::system_ram_mb()
    }

    /// This process's resident memory in MB. Stands in for device memory
    /// on unified-memory (Apple) and CPU-only platforms.
    fn process_resident_memory_mb(&self) -> Option<u64> {
        host::process_resident_memory_mb()
    }
}

/// Detect the platform for this process: first vendor whose management
/// library initializes wins; CPU (system RAM) otherwise. Runs once, at
/// `AppState::new`.
pub(crate) fn detect() -> Arc<dyn Platform> {
    #[cfg(target_os = "linux")]
    if let Some(cuda) = super::cuda::Cuda::probe() {
        return Arc::new(cuda);
    }
    #[cfg(target_os = "macos")]
    if let Some(metal) = super::metal::Metal::probe() {
        return Arc::new(metal);
    }
    Arc::new(super::cpu::Cpu)
}
