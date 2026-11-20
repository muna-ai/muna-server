/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-wide, priority-ordered resource transfers. A resource is keyed by its
//! destination path, not a signed URL; callers share one job and completion.
//! Range parallelism remains inside the SDK downloader. No model task owns the
//! transfer, so retiring one subscriber cannot cancel another model's resource.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use muna::client::DownloadProgressFn;
use tokio::sync::watch;

pub(crate) mod disk;

#[cfg(test)]
mod tests;

/// Shared by all requests from one model preparation, including jobs already
/// queued when that model is promoted from disk to process residency.
#[derive(Clone, Default)]
pub(crate) struct Priority(Arc<AtomicBool>);

impl Priority {
    pub(crate) fn set_process(&self, process: bool) {
        self.0.store(process, Ordering::Relaxed);
    }

    fn is_process(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

type Outcome = Result<(), Arc<str>>;
type Transfer =
    Box<dyn FnOnce(DownloadProgressFn) -> BoxFuture<'static, Result<(), String>> + Send>;

struct Subscriber {
    priority: Priority,
    progress: Option<DownloadProgressFn>,
}

struct Job {
    path: PathBuf,
    subscribers: Mutex<HashMap<u64, Subscriber>>,
    done: watch::Sender<Option<Outcome>>,
    transfer: Mutex<Option<Transfer>>,
}

impl Job {
    fn process(&self) -> bool {
        self.subscribers
            .lock()
            .unwrap()
            .values()
            .any(|s| s.priority.is_process())
    }

    fn report(&self, increment: u64, total: Option<u64>) {
        let callbacks: Vec<_> = self
            .subscribers
            .lock()
            .unwrap()
            .values()
            .filter_map(|s| s.progress.clone())
            .collect();
        // Callbacks may enter the preparation tracker; never run them under
        // the queue or subscriber mutex.
        for callback in callbacks {
            callback(increment, total);
        }
    }
}

#[derive(Default)]
struct Queue {
    jobs: HashMap<PathBuf, Arc<Job>>,
    pending: VecDeque<Arc<Job>>,
    active: usize,
    next_subscriber: u64,
}

#[derive(Default)]
struct Inner {
    queue: Mutex<Queue>,
    disk: disk::DiskBudget,
}

#[derive(Clone, Default)]
pub(crate) struct Downloads(Arc<Inner>);

/// Dropping a requesting future detaches only that caller. A pending job
/// without subscribers is skipped; an active transfer finishes into cache.
struct Subscription {
    job: Arc<Job>,
    id: u64,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.job.subscribers.lock().unwrap().remove(&self.id);
    }
}

impl Downloads {
    pub(crate) fn disk(&self) -> &disk::DiskBudget { &self.0.disk }

    pub(crate) async fn download(
        &self,
        path: &Path,
        priority: Priority,
        progress: Option<DownloadProgressFn>,
        transfer: impl FnOnce(DownloadProgressFn) -> BoxFuture<'static, Result<(), String>>
            + Send
            + 'static,
    ) -> Result<(), String> {
        let (subscription, mut done) = {
            let mut queue = self.0.queue.lock().unwrap();
            let job = match queue.jobs.entry(path.into()) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let (done, _) = watch::channel(None);
                    let job = Arc::new(Job {
                        path: path.into(),
                        subscribers: Mutex::default(),
                        done,
                        transfer: Mutex::new(Some(Box::new(transfer))),
                    });
                    entry.insert(job.clone());
                    queue.pending.push_back(job.clone());
                    job
                }
            };
            let id = queue.next_subscriber;
            queue.next_subscriber = queue.next_subscriber.wrapping_add(1);
            job.subscribers
                .lock()
                .unwrap()
                .insert(id, Subscriber { priority, progress });
            let done = job.done.subscribe();
            (Subscription { job, id }, done)
        };
        self.dispatch();
        loop {
            if let Some(outcome) = done.borrow_and_update().clone() {
                drop(subscription);
                return outcome.map_err(|error| error.to_string());
            }
            done.changed()
                .await
                .map_err(|_| "Resource download stopped.".to_string())?;
        }
    }

    fn dispatch(&self) {
        let mut queue = self.0.queue.lock().unwrap();
        while queue.active == 0 {
            // Dynamic priority lets a process load promote an existing job.
            // FIFO is retained within each priority; active files are not preempted.
            let index = queue
                .pending
                .iter()
                .position(|job| job.process())
                .unwrap_or(0);
            let Some(job) = queue.pending.remove(index) else {
                break;
            };
            if job.subscribers.lock().unwrap().is_empty() {
                queue.jobs.remove(&job.path);
                continue;
            }
            queue.active += 1;
            let scheduler = self.clone();
            tokio::spawn(async move {
                // A separate join boundary converts an unexpected transfer panic
                // into a failure for every waiter and releases the queue slot.
                let active = job.clone();
                scheduler.0.disk.started(&job.path);
                let outcome = tokio::spawn(async move {
                    if tokio::fs::metadata(&active.path)
                        .await
                        .is_ok_and(|m| m.is_file())
                    {
                        return Ok(());
                    }
                    let transfer = active.transfer.lock().unwrap().take().unwrap();
                    let observed = active.clone();
                    let callback: DownloadProgressFn =
                        Arc::new(move |increment, total| observed.report(increment, total));
                    transfer(callback).await
                })
                .await
                .unwrap_or_else(|_| Err("Resource download task failed.".into()));
                scheduler.0.disk.finished(&job.path, outcome.is_ok());
                {
                    let mut queue = scheduler.0.queue.lock().unwrap();
                    job.done
                        .send_replace(Some(outcome.map_err(Arc::<str>::from)));
                    queue.jobs.remove(&job.path);
                    queue.active -= 1;
                }
                scheduler.dispatch();
            });
        }
    }
}
