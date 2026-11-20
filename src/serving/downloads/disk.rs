/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Admission for the actual manifest selected on this node. Reservations are
//! shared by cache path, independent of deployment credentials and subscribers.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

#[derive(Default)]
struct Resource {
    size: u64,
    written: u64,
    users: usize,
    active: bool,
}

type Available = dyn Fn(&Path) -> Option<u64> + Send + Sync;

struct Inner {
    resources: Mutex<HashMap<PathBuf, Resource>>,
    available: Box<Available>,
    changed: Notify,
}

#[derive(Clone)]
pub(crate) struct DiskBudget(Arc<Inner>);

pub(crate) struct Reservation {
    budget: DiskBudget,
    paths: Vec<PathBuf>,
}

impl Default for DiskBudget {
    fn default() -> Self {
        Self::new(|root| crate::platform::disk_space_bytes(root).map(|(free, _)| free))
    }
}

impl DiskBudget {
    fn new(available: impl Fn(&Path) -> Option<u64> + Send + Sync + 'static) -> Self {
        Self(Arc::new(Inner {
            resources: Mutex::default(),
            available: Box::new(available),
            changed: Notify::new(),
        }))
    }

    /// Resolve sizes before entering here. The lock covers both the filesystem
    /// observation and the reservation so simultaneous preparations cannot
    /// spend the same free bytes. Already-written bytes are in statvfs already.
    pub(crate) fn try_reserve(
        &self,
        root: &Path,
        manifest: &HashMap<PathBuf, u64>,
    ) -> Result<Option<Reservation>, String> {
        let mut resources = self.0.resources.lock().unwrap();
        let mut missing = Vec::new();
        let mut additional = 0u64;
        for (path, size) in manifest {
            match std::fs::metadata(path) {
                Ok(meta) if meta.is_file() && meta.len() == *size => continue,
                Ok(_) => return Err("Cached resource has an unexpected size or type.".into()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(format!("Cannot inspect cached resource: {error}")),
            }
            if let Some(resource) = resources.get(path) {
                if resource.size != *size {
                    return Err("Conflicting sizes for one cached resource.".into());
                }
            } else {
                additional = additional
                    .checked_add(*size)
                    .ok_or("Resource sizes overflowed.")?;
            }
            missing.push((path.clone(), *size));
        }
        let pending = resources
            .values()
            .try_fold(0u64, |sum, resource| {
                sum.checked_add(resource.size.saturating_sub(resource.written))
            })
            .ok_or("Reserved resource sizes overflowed.")?;
        if additional > 0 {
            let free =
                (self.0.available)(root).ok_or("Cannot inspect available cache disk space.")?;
            if additional > free.saturating_sub(pending) {
                return Ok(None);
            }
        }
        let mut paths = Vec::new();
        for (path, size) in missing {
            let resource = resources.entry(path.clone()).or_insert(Resource {
                size,
                ..Default::default()
            });
            resource.users += 1;
            paths.push(path);
        }
        Ok(Some(Reservation {
            budget: self.clone(),
            paths,
        }))
    }

    pub(crate) async fn reserve(
        &self,
        root: &Path,
        manifest: &HashMap<PathBuf, u64>,
        waiting: impl Fn(bool),
    ) -> Result<Reservation, String> {
        loop {
            // Subscribe before inspecting capacity: completion/cancellation
            // between the check and await must not strand a preparation.
            let changed = self.0.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(reservation) = self.try_reserve(root, manifest)? {
                waiting(false);
                return Ok(reservation);
            }
            waiting(true);
            changed.await;
        }
    }

    pub(crate) fn started(&self, path: &Path) {
        if let Some(resource) = self.0.resources.lock().unwrap().get_mut(path) {
            resource.active = true;
            resource.written = 0;
        }
    }

    pub(crate) fn advance(&self, path: &Path, bytes: u64) {
        if let Some(resource) = self.0.resources.lock().unwrap().get_mut(path) {
            resource.written = resource.written.saturating_add(bytes).min(resource.size);
        }
    }

    pub(crate) fn finished(&self, path: &Path, success: bool) {
        let mut resources = self.0.resources.lock().unwrap();
        if let Some(resource) = resources.get_mut(path) {
            resource.active = false;
            resource.written = if success { resource.size } else { 0 };
            if resource.users == 0 {
                resources.remove(path);
            }
        }
        drop(resources);
        self.wake();
    }

    /// Also called by periodic node reporting, to notice capacity freed by
    /// actors outside the downloader without resolving metadata again.
    pub(crate) fn wake(&self) {
        self.0.changed.notify_waiters();
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut resources = self.budget.0.resources.lock().unwrap();
        for path in &self.paths {
            if let Some(resource) = resources.get_mut(path) {
                resource.users -= 1;
                // An active transfer outlives its last subscriber and must
                // retain its reservation until its file has been published.
                if resource.users == 0 && !resource.active {
                    resources.remove(path);
                }
            }
        }
        drop(resources);
        self.budget.wake();
    }
}

#[cfg(test)]
mod tests;
