/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Null platform for tests: no devices, nothing to warm, no memory
//! accounting, and fixed host figures so heartbeat payloads are
//! deterministic.

use std::path::Path;

use super::family::GpuVendor;
use super::platform::{DeviceMetrics, Platform, WarmReport};

pub(crate) struct NullPlatform;

impl NullPlatform {
    pub(crate) const DISK_FREE_MB: u64 = 500_000;
    pub(crate) const DISK_TOTAL_MB: u64 = 1_000_000;
    pub(crate) const SYSTEM_RAM_MB: u64 = 65_536;
    pub(crate) const PROCESS_RSS_MB: u64 = 512;
}

impl Platform for NullPlatform {

    fn vendor(&self) -> GpuVendor {
        GpuVendor::Unknown
    }

    fn warm(&self) -> Result<Option<WarmReport>, String> {
        Ok(None)
    }

    fn devices(&self) -> Vec<DeviceMetrics> {
        Vec::new()
    }

    fn memory_used_mb(&self) -> Option<u64> {
        None
    }

    fn disk_space_mb(&self, _path: &Path) -> Option<(u64, u64)> {
        Some((Self::DISK_FREE_MB, Self::DISK_TOTAL_MB))
    }

    fn system_ram_mb(&self) -> Option<u64> {
        Some(Self::SYSTEM_RAM_MB)
    }

    fn process_resident_memory_mb(&self) -> Option<u64> {
        Some(Self::PROCESS_RSS_MB)
    }
}
