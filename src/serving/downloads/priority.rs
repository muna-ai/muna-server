/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Urgency follows a model preparation, including shared jobs already queued.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::serving::preparation::PreparationTracker;

#[derive(Default)]
struct State {
    process: AtomicBool,
    deployment: AtomicBool,
    model: Mutex<Option<(String, PreparationTracker)>>,
}

#[derive(Clone, Default)]
pub(crate) struct Priority(Arc<State>);

impl Priority {
    pub(crate) fn set_process(&self, process: bool) {
        self.0.process.store(process, Ordering::Relaxed);
    }

    pub(crate) fn set_deployment(&self, deployment: bool) {
        self.0.deployment.store(deployment, Ordering::Relaxed);
    }

    pub(crate) fn track(&self, tag: &str, tracker: PreparationTracker) {
        *self.0.model.lock().unwrap() = Some((tag.into(), tracker));
    }

    pub(super) fn rank(&self) -> (u8, Option<u64>) {
        let class = if self.0.process.load(Ordering::Relaxed) {
            2
        } else if self.0.deployment.load(Ordering::Relaxed) {
            1
        } else {
            0
        };
        let remaining = self
            .0
            .model
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|(tag, tracker)| tracker.remaining_bytes(tag));
        (class, remaining)
    }
}
