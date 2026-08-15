/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Prediction serving: model lifecycle (registry), batching-aware dispatch,
//! blocking FFI execution, and per-model stats. A request first goes through
//! the registry (ensure the model is loaded), then the dispatcher (apply the
//! model's batch plan), which executes through `predict`.

pub(crate) mod batch;
pub(crate) mod dispatch;
pub(crate) mod predict;
pub(crate) mod registry;
pub(crate) mod stats;
