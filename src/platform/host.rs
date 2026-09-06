/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Host metrics shared by every vendor backend: disk space, system RAM,
//! process resident memory. These vary by OS, not by vendor, so they back
//! the `Platform` trait's default methods rather than living in any one
//! backend.

/// Free and total space in MB of the filesystem containing `path`, walking
/// up to the nearest existing ancestor.
pub(super) fn disk_space_mb(path: &std::path::Path) -> Option<(u64, u64)> {
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

/// Total physical RAM in MB.
pub(super) fn system_ram_mb() -> Option<u64> {
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

/// Process resident memory in MB.
pub(super) fn process_resident_memory_mb() -> Option<u64> {
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
    const INFO_COUNT: u32 = (
        std::mem::size_of::<MachTaskBasicInfo>() /
        std::mem::size_of::<u32>()
    ) as u32;
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
