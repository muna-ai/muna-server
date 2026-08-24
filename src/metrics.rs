/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! GPU and memory metrics collection (NVML on Linux, Metal on macOS, system
//! RAM fallback on CPU-only nodes).

use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(dead_code)]
pub(crate) enum GpuVendor {
    Nvidia,
    Apple,
    Amd,
    Unknown,
}

/// Standardized GPU family, the vocabulary shared with `muna deploy --gpu`
/// and the control plane's capacity records. Wire spelling is lowercase
/// (`"h100"`, `"b200"`). Mirrored by the control plane, which additionally 
/// tolerates unknown slugs.
#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub(crate) enum GpuFamily {
    Cpu,
    A10G,
    L4,
    L40S,
    A100,
    H100,
    B200,
    MI350X,
    MI355X,
}

impl GpuFamily {

    /// Derive the family from a device name as reported by NVML / ROCm
    /// (e.g. "NVIDIA H100 80GB HBM3" -> `H100`). Matches whole
    /// alphanumeric tokens, not substrings, so "NVIDIA L40S" never maps
    /// to `L4`. `None` for devices outside the fleet vocabulary; the
    /// control plane then falls back to raw-name matching.
    pub fn from_device_name(name: &str) -> Option<GpuFamily> {
        let tokens: Vec<String> = name
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_uppercase)
            .collect();
        let has = |token: &str| tokens.iter().any(|t| t == token);
        [
            ("A10G",   GpuFamily::A10G),
            ("L40S",   GpuFamily::L40S),
            ("L4",     GpuFamily::L4),
            ("A100",   GpuFamily::A100),
            ("H100",   GpuFamily::H100),
            ("B200",   GpuFamily::B200),
            ("MI350X", GpuFamily::MI350X),
            ("MI355X", GpuFamily::MI355X),
        ]
        .into_iter()
        .find_map(|(token, family)| has(token).then_some(family))
    }
}

#[derive(Serialize)]
pub(crate) struct GpuMetrics {
    /// Device index (0-based).
    pub index: u32,
    /// GPU vendor (nvidia, apple, amd).
    pub vendor: GpuVendor,
    /// GPU device name (e.g. "NVIDIA H100", "Apple M4 Pro").
    pub name: String,
    /// Standardized GPU family derived from the device name; `None` for
    /// devices outside the fleet vocabulary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<GpuFamily>,
    /// Total VRAM in MB. On Apple Silicon, this is the recommended max working set size.
    pub vram_total_mb: u64,
    /// Currently allocated VRAM in MB. `None` if the backend cannot report this.
    pub vram_used_mb: Option<u64>,
    /// Free VRAM in MB. `None` if the backend cannot report this.
    pub vram_free_mb: Option<u64>,
    /// GPU compute utilization as a percentage (0-100). `None` on Apple Silicon.
    pub utilization_pct: Option<u32>,
    /// GPU temperature in degrees Celsius. `None` on Apple Silicon.
    pub temperature_c: Option<u32>,
}

/// Free and total space in MB of the filesystem containing `path`, walking
/// up to the nearest existing ancestor. Feeds capacity reporting: resources
/// are never deleted, so the control plane must see remaining disk to know
/// when to stop placing models on this node.
pub(crate) fn disk_space_mb(path: &std::path::Path) -> Option<(u64, u64)> {
    path.ancestors().find_map(statvfs_mb)
}

#[cfg(unix)]
fn statvfs_mb(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let cpath = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(cpath.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    let frsize = if stat.f_frsize > 0 { stat.f_frsize } else { stat.f_bsize } as u64;
    let to_mb = |blocks: u64| blocks.saturating_mul(frsize) / (1024 * 1024);
    Some((to_mb(stat.f_bavail as u64), to_mb(stat.f_blocks as u64)))
}

#[cfg(not(unix))]
fn statvfs_mb(_path: &std::path::Path) -> Option<(u64, u64)> {
    None
}

pub(crate) fn collect_gpu_metrics() -> Vec<GpuMetrics> {
    let mut metrics = Vec::new();
    #[cfg(target_os = "linux")]
    metrics.extend(collect_nvml_metrics());
    #[cfg(target_os = "macos")]
    metrics.extend(collect_metal_metrics());
    if metrics.is_empty() {
        if let Some(total_mb) = system_ram_mb() {
            let used_mb = process_resident_memory_mb();
            metrics.push(GpuMetrics {
                index: 0,
                vendor: GpuVendor::Unknown,
                name: "CPU (system RAM)".into(),
                family: Some(GpuFamily::Cpu),
                vram_total_mb: total_mb,
                vram_used_mb: used_mb,
                vram_free_mb: used_mb.map(|u| total_mb.saturating_sub(u)),
                utilization_pct: None,
                temperature_c: None,
            });
        }
    }
    metrics
}

/// Total memory used in MB -- GPU VRAM if available, otherwise process
/// resident memory as a proxy for CPU-only instances. Used for measuring a
/// model's memory footprint as a load-time delta.
pub(crate) fn total_memory_used_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        if let Some(total) = nvml_vram_used_mb() {
            return Some(total);
        }
    }
    process_resident_memory_mb()
}

#[cfg(target_os = "linux")]
fn nvml_vram_used_mb() -> Option<u64> {
    let nvml = init_nvml().ok()?;
    let count = nvml.device_count().ok().filter(|&c| c > 0)?;
    let mut total = 0u64;
    for i in 0..count {
        if let Ok(device) = nvml.device_by_index(i) {
            if let Ok(mem) = device.memory_info() {
                total += mem.used / (1024 * 1024);
            }
        }
    }
    Some(total)
}

#[cfg(target_os = "linux")]
static NVML_LIB_PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();

#[cfg(target_os = "linux")]
fn find_nvml_lib() -> Option<std::path::PathBuf> {
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

#[cfg(target_os = "linux")]
pub(crate) fn init_nvml() -> Result<nvml_wrapper::Nvml, nvml_wrapper::error::NvmlError> {
    nvml_wrapper::Nvml::init().or_else(|first_err| {
        let cached = NVML_LIB_PATH.get_or_init(find_nvml_lib);
        match cached {
            Some(path) => nvml_wrapper::Nvml::builder()
                .lib_path(path.as_os_str())
                .init(),
            None => Err(first_err),
        }
    })
}

#[cfg(target_os = "linux")]
fn collect_nvml_metrics() -> Vec<GpuMetrics> {
    let nvml = match init_nvml() {
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("NVML init failed: {e}");
            return Vec::new();
        }
    };
    let count = nvml.device_count().unwrap_or(0);
    (0..count)
        .filter_map(|i| {
            let device = nvml.device_by_index(i).ok()?;
            let name = device.name().unwrap_or_else(|_| "unknown".into());
            let mem = device.memory_info().ok()?;
            let util = device.utilization_rates().ok();
            let temp = device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok();
            Some(GpuMetrics {
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

#[cfg(target_os = "macos")]
fn collect_metal_metrics() -> Vec<GpuMetrics> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    let device = match MTLCreateSystemDefaultDevice() {
        Some(d) => d,
        None => {
            tracing::warn!("no Metal device found");
            return Vec::new();
        }
    };
    let name = device.name().to_string();
    let vram_total_mb = device.recommendedMaxWorkingSetSize() / (1024 * 1024);
    let vram_used_mb = process_resident_memory_mb();
    vec![GpuMetrics {
        index: 0,
        vendor: GpuVendor::Apple,
        family: None,
        name,
        vram_total_mb,
        vram_used_mb,
        vram_free_mb: vram_used_mb.map(|used| vram_total_mb.saturating_sub(used)),
        utilization_pct: None,
        temperature_c: None,
    }]
}

/// Total physical RAM in MB.
fn system_ram_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: u64 = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
                return Some(kb / 1024);
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        extern "C" {
            fn sysctl(
                name: *const i32, namelen: u32,
                oldp: *mut u8, oldlenp: *mut usize,
                newp: *const u8, newlen: usize,
            ) -> i32;
        }
        let mib: [i32; 2] = [
            6,  // CTL_HW
            24, // HW_MEMSIZE
        ];
        let mut memsize: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let ret = unsafe {
            sysctl(
                mib.as_ptr(), 2,
                &mut memsize as *mut u64 as *mut u8, &mut len,
                std::ptr::null(), 0,
            )
        };
        if ret == 0 { Some(memsize / (1024 * 1024)) } else { None }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// Process resident memory in MB. Used as a fallback for VRAM on CPU-only nodes.
fn process_resident_memory_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = 4096u64;
        Some(pages * page_size / (1024 * 1024))
    }
    #[cfg(target_os = "macos")]
    {
        process_resident_memory_mb_macos()
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_maps_real_device_names() {
        let cases = [
            ("NVIDIA H100 80GB HBM3",     Some(GpuFamily::H100)),
            ("NVIDIA H100 PCIe",          Some(GpuFamily::H100)),
            ("NVIDIA A100-SXM4-80GB",     Some(GpuFamily::A100)),
            ("NVIDIA A10G",               Some(GpuFamily::A10G)),
            ("NVIDIA B200",               Some(GpuFamily::B200)),
            ("AMD Instinct MI355X",       Some(GpuFamily::MI355X)),
            ("Apple M4 Pro",              None),
            ("Tesla T4",                  None),
        ];
        for (name, expected) in cases {
            assert_eq!(GpuFamily::from_device_name(name), expected, "{name}");
        }
    }

    /// "L4" is a substring of "L40S"; token matching keeps them distinct.
    #[test]
    fn family_does_not_confuse_l4_with_l40s() {
        assert_eq!(GpuFamily::from_device_name("NVIDIA L4"), Some(GpuFamily::L4));
        assert_eq!(GpuFamily::from_device_name("NVIDIA L40S"), Some(GpuFamily::L40S));
    }

    /// The wire spelling is the lowercase slug shared with `--gpu`.
    #[test]
    fn family_wire_spelling_is_lowercase() {
        assert_eq!(serde_json::to_value(GpuFamily::H100).unwrap(), "h100");
        assert_eq!(serde_json::to_value(GpuFamily::MI355X).unwrap(), "mi355x");
    }
}

/// Query this process's physical memory footprint via mach task_info.
/// On Apple Silicon with unified memory, this includes GPU allocations.
#[cfg(target_os = "macos")]
fn process_resident_memory_mb_macos() -> Option<u64> {
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: u32 = 20;
    const INFO_COUNT: u32 =
        (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: u32, info: *mut u8, count: *mut u32) -> i32;
    }
    unsafe {
        let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
        let mut count = INFO_COUNT;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut u8,
            &mut count,
        );
        if kr == 0 {
            Some(info.assume_init().resident_size / (1024 * 1024))
        } else {
            None
        }
    }
}
