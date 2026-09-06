/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! NVIDIA: NVML for device metrics (one handle for the process lifetime),
//! libcuda for the driver warm.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;
use libloading::{Library, Symbol};
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

use super::family::{GpuFamily, GpuVendor};
use super::platform::{DeviceMetrics, Platform, WarmReport};

pub(crate) struct Cuda {
    nvml: Nvml,
    /// Set by `warm`; holding the library keeps libcuda (and the ~250 MB
    /// of driver libraries it pulls in) mapped so their pages stay
    /// resident, and the retained primary contexts are the ones every
    /// engine's first `cudaSetDevice` would otherwise create.
    driver: OnceLock<Library>,
}

impl Cuda {

    /// `Some` when NVML initializes; `None` on hosts without an NVIDIA
    /// driver.
    pub(crate) fn probe() -> Option<Self> {
        match init_nvml() {
            Ok(nvml) => Some(Self { nvml, driver: OnceLock::new() }),
            Err(e) => {
                tracing::debug!(error = %e, "NVML unavailable; not an NVIDIA host");
                None
            }
        }
    }
}

impl Platform for Cuda {

    fn vendor(&self) -> GpuVendor {
        GpuVendor::Nvidia
    }

    fn warm(&self) -> Result<Option<WarmReport>, String> {
        if self.driver.get().is_some() {
            return Ok(None);
        }
        // SAFETY: libcuda.so.1 is the NVIDIA driver's user-space library;
        // loading it runs only its constructors, which is how every CUDA
        // process starts.
        let lib = unsafe { Library::new("libcuda.so.1") }.map_err(|e| e.to_string())?;
        // SAFETY: the symbol signatures match the CUDA driver API
        // (cuda.h); all out-pointers are valid for the duration of the
        // call.
        let devices = unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> c_int> =
                lib.get(b"cuInit\0").map_err(|e| e.to_string())?;
            let device_count: Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> =
                lib.get(b"cuDeviceGetCount\0").map_err(|e| e.to_string())?;
            let device_get: Symbol<unsafe extern "C" fn(*mut c_int, c_int) -> c_int> =
                lib.get(b"cuDeviceGet\0").map_err(|e| e.to_string())?;
            let ctx_retain: Symbol<unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int> =
                lib.get(b"cuDevicePrimaryCtxRetain\0").map_err(|e| e.to_string())?;
            check(cu_init(0), "cuInit")?;
            let mut count: c_int = 0;
            check(device_count(&mut count), "cuDeviceGetCount")?;
            for index in 0..count {
                let mut device: c_int = 0;
                check(device_get(&mut device, index), "cuDeviceGet")?;
                let mut ctx: *mut c_void = std::ptr::null_mut();
                // Never released on purpose: see `driver` doc.
                check(ctx_retain(&mut ctx, device), "cuDevicePrimaryCtxRetain")?;
            }
            count.max(0) as usize
        };
        let _ = self.driver.set(lib);
        Ok(Some(WarmReport { devices }))
    }

    fn devices(&self) -> Vec<DeviceMetrics> {
        let count = self.nvml.device_count().unwrap_or(0);
        (0..count)
            .filter_map(|i| {
                let device = self.nvml.device_by_index(i).ok()?;
                let name = device.name().unwrap_or_else(|_| "unknown".into());
                let mem = device.memory_info().ok()?;
                let util = device.utilization_rates().ok();
                let temp = device.temperature(TemperatureSensor::Gpu).ok();
                Some(DeviceMetrics {
                    index: i,
                    vendor: GpuVendor::Nvidia,
                    family: GpuFamily::from_device_name(&name),
                    name,
                    vram_total_mb: mem.total / (1024 * 1024),
                    vram_used_mb: Some(mem.used / (1024 * 1024)),
                    vram_free_mb: Some(mem.free / (1024 * 1024)),
                    utilization_pct: util.map(|u| u.gpu),
                    temperature_c: temp,
                })
            })
            .collect()
    }

    fn memory_used_mb(&self) -> Option<u64> {
        let count = self.nvml.device_count().ok().filter(|&c| c > 0)?;
        let mut total = 0u64;
        for i in 0..count {
            if let Ok(device) = self.nvml.device_by_index(i) {
                if let Ok(mem) = device.memory_info() {
                    total += mem.used / (1024 * 1024);
                }
            }
        }
        Some(total)
    }
}

fn check(
    code: c_int,
    what: &str
) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{what} returned CUDA error {code}"))
    }
}

static NVML_LIB_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Initialize NVML, falling back to a filesystem search for the library
/// when the default loader path misses (containers without ldconfig).
fn init_nvml() -> Result<Nvml, NvmlError> {
    Nvml::init().or_else(|first_err| {
        let cached = NVML_LIB_PATH.get_or_init(find_nvml_lib);
        match cached {
            Some(path) => Nvml::builder()
                .lib_path(path.as_os_str())
                .init(),
            None => Err(first_err),
        }
    })
}

fn find_nvml_lib() -> Option<PathBuf> {
    let search_dirs = [
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/local/cuda/lib64",
    ];
    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("libnvidia-ml.so") {
                    let path = entry.path();
                    tracing::info!(path = %path.display(), "found NVML library via search");
                    return Some(path);
                }
            }
        }
    }
    None
}
