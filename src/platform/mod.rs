/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Platform abstraction: one detected platform per process, owning its
//! vendor handles for the process lifetime. Three vendor concerns
//! (detect, warm the driver, report devices) plus the host metrics every
//! vendor shares (disk, RAM, process memory), so a new backend (ROCm/HIP,
//! ...) is one file implementing one trait.

mod cpu;
#[cfg(target_os = "linux")]
mod cuda;
mod family;
mod host;
#[cfg(target_os = "macos")]
mod metal;
#[cfg(test)]
mod null;
mod platform;

#[cfg(test)]
pub(crate) use null::NullPlatform;
pub(crate) use platform::{detect, DeviceMetrics, Platform, WarmReport};
