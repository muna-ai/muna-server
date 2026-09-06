/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! CPU-only fallback: system RAM stands in for device memory, process
//! resident memory for usage. Nothing to warm.

use super::family::{GpuFamily, GpuVendor};
use super::platform::{DeviceMetrics, Platform, WarmReport};

pub(crate) struct Cpu;

impl Platform for Cpu {

    fn vendor(&self) -> GpuVendor {
        GpuVendor::Unknown
    }

    fn warm(&self) -> Result<Option<WarmReport>, String> {
        Ok(None)
    }

    fn devices(&self) -> Vec<DeviceMetrics> {
        let Some(total_mb) = self.system_ram_mb() else {
            return Vec::new();
        };
        let used_mb = self.process_resident_memory_mb();
        vec![DeviceMetrics {
            index: 0,
            vendor: GpuVendor::Unknown,
            name: "CPU (system RAM)".into(),
            family: Some(GpuFamily::Cpu),
            vram_total_mb: total_mb,
            vram_used_mb: used_mb,
            vram_free_mb: used_mb.map(|u| total_mb.saturating_sub(u)),
            utilization_pct: None,
            temperature_c: None,
        }]
    }

    fn memory_used_mb(&self) -> Option<u64> {
        self.process_resident_memory_mb()
    }
}
