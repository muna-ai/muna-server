/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Apple: the system default Metal device, created once at probe and kept
//! for the process lifetime. Unified memory means process resident memory
//! is the best available "VRAM used" figure.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use super::family::{GpuFamily, GpuVendor};
use super::platform::{DeviceMetrics, Platform, WarmReport};

pub(crate) struct Metal {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl Metal {

    /// `Some` when a Metal device exists; `None` on headless / non-Apple
    /// hardware.
    pub(crate) fn probe() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        Some(Self { device })
    }
}

impl Platform for Metal {

    fn vendor(&self) -> GpuVendor {
        GpuVendor::Apple
    }

    /// Creating the default device IS the warm on Metal, and `probe`
    /// already did it.
    fn warm(&self) -> Result<Option<WarmReport>, String> {
        Ok(None)
    }

    fn devices(&self) -> Vec<DeviceMetrics> {
        let name = self.device.name().to_string();
        let vram_total_mb = self.device.recommendedMaxWorkingSetSize() / (1024 * 1024);
        let vram_used_mb = self.process_resident_memory_mb();
        vec![DeviceMetrics {
            index: 0,
            vendor: GpuVendor::Apple,
            // Apple devices are outside the fleet vocabulary today; this
            // resolves to `None` until the vocabulary grows.
            family: GpuFamily::from_device_name(&name),
            name,
            vram_total_mb,
            vram_used_mb,
            vram_free_mb: vram_used_mb.map(|used| vram_total_mb.saturating_sub(used)),
            utilization_pct: None,
            temperature_c: None,
        }]
    }

    fn memory_used_mb(&self) -> Option<u64> {
        self.process_resident_memory_mb()
    }
}
