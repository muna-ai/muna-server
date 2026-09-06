/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! In-process wakeups between subsystems. One named channel per concern;
//! add a field rather than generalizing into an event bus: `Notify`'s
//! single stored permit is exactly the coalescing the consumers rely on,
//! and a future signal that needs payloads or fan-out gets its own field
//! with its own type.

use tokio::sync::Notify;

#[derive(Default)]
pub(crate) struct NotificationCenter {
    /// Reported node state changed: a model's lifecycle (Loading, Ready,
    /// Failed, unloaded), its cache tier (Caching, Cached, Failed), or the
    /// drain flag. Awaited by the control-plane heartbeat loop so a
    /// transition is reported immediately instead of on the next tick.
    /// `notify_one` stores at most one permit, so any number of changes
    /// during an in-flight beat collapse into exactly one follow-up beat.
    pub status: Notify,
}

impl NotificationCenter {

    /// Signal that the node's reported state changed. Call after every
    /// write to a model's lifecycle or cache tier, or to the drain flag;
    /// the heartbeat loop wakes and reports the new state immediately.
    /// Cheap and coalescing (one stored permit), so callers need not
    /// batch or deduplicate.
    pub(crate) fn status_changed(&self) {
        self.status.notify_one();
    }
}
