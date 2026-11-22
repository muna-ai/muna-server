/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Node-wide resource transfers. Complete files bypass the queue; missing files
//! share one resumable job. Each bounded range batch yields to model urgency.

use std::cmp::Reverse;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use muna::DownloadProgressFn;
use tokio::sync::watch;

use crate::serving::preparation::PreparationTracker;

pub(crate) mod disk;
mod priority;
mod progress;
mod transfer;

pub(crate) use priority::Priority;
use progress::DownloadProgress;
use transfer::{Limits, Transfer};

#[cfg(test)]
mod tests;

// Large models still receive turns when small preparations keep arriving.
// Aging applies within an urgency class only.
const MAX_WAIT: Duration = Duration::from_secs(30);
type Outcome = Result<(), Arc<str>>;

struct Subscriber {
    priority: Priority,
    progress: Option<DownloadProgressFn>,
}

struct State {
    subscribers: HashMap<u64, Subscriber>,
    available: u64,
    total: Option<u64>,
    last_served: Instant,
}

struct Job {
    path: PathBuf,
    state: Mutex<State>,
    done: watch::Sender<Option<Outcome>>,
    transfer: tokio::sync::Mutex<Transfer>,
    tracker: Option<PreparationTracker>,
    presentation: Mutex<Option<DownloadProgress>>,
    draw_progress: bool,
    started: AtomicBool,
    queued: Instant,
    order: u64,
}

impl Job {
    fn rank(&self, now: Instant) -> (Reverse<u8>, bool, u64, u64) {
        let state = self.state.lock().unwrap();
        let class = state
            .subscribers
            .values()
            .map(|s| s.priority.rank().0)
            .max()
            .unwrap_or(0);
        let remaining = state
            .subscribers
            .values()
            .filter_map(|subscriber| {
                let (tier, remaining) = subscriber.priority.rank();
                (tier == class).then_some(remaining).flatten()
            })
            .min()
            .or_else(|| {
                state
                    .total
                    .map(|total| total.saturating_sub(state.available))
            })
            .unwrap_or(u64::MAX);
        let wait = now.saturating_duration_since(state.last_served);
        let overdue = wait >= MAX_WAIT;
        let distance = if overdue {
            u64::MAX - wait.as_millis().min(u64::MAX as u128) as u64
        } else {
            remaining
        };
        (Reverse(class), !overdue, distance, self.order)
    }

    fn report(&self, disk: &disk::DiskBudget, increment: u64, total: Option<u64>) {
        let callbacks: Vec<_> = {
            let mut state = self.state.lock().unwrap();
            state.total = total.or(state.total);
            state.available = state.available.saturating_add(increment);
            state
                .subscribers
                .values()
                .filter_map(|s| s.progress.clone())
                .collect()
        };
        // Physical bytes count once. Every model's tracker refers to this same
        // shared resource. Subscriber callbacks run outside all queue locks.
        disk.advance(&self.path, increment);
        if let Some(tracker) = &self.tracker {
            tracker.advance(&self.path, increment, total);
        }
        if let Some(presentation) = self.presentation.lock().unwrap().as_ref() {
            presentation.advance(increment, total);
        }
        for callback in callbacks {
            callback(increment, total);
        }
    }
}

#[derive(Default)]
struct Queue {
    jobs: HashMap<PathBuf, Arc<Job>>,
    pending: VecDeque<Arc<Job>>,
    running: bool,
    next_subscriber: u64,
}

#[derive(Default)]
struct Inner {
    queue: Mutex<Queue>,
    disk: disk::DiskBudget,
    limits: Limits,
}

#[derive(Clone, Default)]
pub(crate) struct Downloads(Arc<Inner>);

/// Detaching a model never cancels another subscriber's resource. Once a file
/// starts, it finishes into cache even if its original deployment is retired.
struct Subscription {
    job: Arc<Job>,
    id: u64,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.job.state.lock().unwrap().subscribers.remove(&self.id);
    }
}

impl Downloads {
    pub(crate) fn disk(&self) -> &disk::DiskBudget {
        &self.0.disk
    }

    pub(crate) async fn download(
        &self,
        http: &reqwest::Client,
        url: &str,
        path: &Path,
        priority: Priority,
        progress: Option<DownloadProgressFn>,
        tracker: Option<PreparationTracker>,
    ) -> Result<(), String> {
        // A newly deployed model can link existing resources immediately,
        // even while an unrelated resource is downloading.
        if let Ok(metadata) = tokio::fs::metadata(path).await {
            if metadata.is_file() {
                if let Some(progress) = progress {
                    progress(metadata.len(), Some(metadata.len()));
                }
                return Ok(());
            }
        }
        let (subscription, mut done, available, total) = {
            let mut queue = self.0.queue.lock().unwrap();
            let order = queue.next_subscriber;
            let new_job = !queue.jobs.contains_key(path);
            let job = queue
                .jobs
                .entry(path.into())
                .or_insert_with(|| {
                    let (done, _) = watch::channel(None);
                    Arc::new(Job {
                        path: path.into(),
                        state: Mutex::new(State {
                            subscribers: HashMap::new(),
                            available: 0,
                            total: None,
                            last_served: Instant::now(),
                        }),
                        done,
                        transfer: tokio::sync::Mutex::new(
                            Transfer::new(http.clone(), url.into(), path).with_size(
                                tracker
                                    .as_ref()
                                    .and_then(|tracker| tracker.resource_size(path)),
                            ),
                        ),
                        tracker,
                        presentation: Mutex::default(),
                        draw_progress: progress.is_none(),
                        started: AtomicBool::new(false),
                        queued: Instant::now(),
                        order,
                    })
                })
                .clone();
            if new_job {
                queue.pending.push_back(job.clone());
            }
            let id = queue.next_subscriber;
            queue.next_subscriber = queue.next_subscriber.wrapping_add(1);
            let (available, total) = {
                let mut state = job.state.lock().unwrap();
                state.subscribers.insert(
                    id,
                    Subscriber {
                        priority,
                        progress: progress.clone(),
                    },
                );
                (state.available, state.total)
            };
            let done = job.done.subscribe();
            (Subscription { job, id }, done, available, total)
        };
        // Attaching waiters receive bytes already transferred exactly once.
        if let Some(progress) = progress.filter(|_| available > 0 || total.is_some()) {
            progress(available, total);
        }
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
        if queue.running {
            return;
        }
        queue.running = true;
        let scheduler = self.clone();
        tokio::spawn(async move {
            scheduler.run().await;
        });
    }

    async fn run(&self) {
        loop {
            let job = {
                let mut queue = self.0.queue.lock().unwrap();
                let now = Instant::now();
                let Some(index) = queue
                    .pending
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, job)| job.rank(now))
                    .map(|(index, _)| index)
                else {
                    queue.running = false;
                    return;
                };
                let job = queue.pending.remove(index).unwrap();
                if !job.started.load(Ordering::Relaxed)
                    && job.state.lock().unwrap().subscribers.is_empty()
                {
                    queue.jobs.remove(&job.path);
                    continue;
                }
                job
            };
            // A join boundary keeps panicking callbacks from stranding the
            // worker and every unrelated resource behind the current job.
            let active = job.clone();
            let scheduler = self.clone();
            if tokio::spawn(async move {
                scheduler.advance(&active).await;
            })
            .await
            .is_err()
            {
                self.finish(&job, Err("Resource download task failed.".into()))
                    .await;
            }
            // Let SDK callers enqueue their next dependency. This is a
            // cooperative yield, never a timer or readiness poll.
            tokio::task::yield_now().await;
        }
    }

    async fn advance(&self, job: &Arc<Job>) {
        if !job.started.swap(true, Ordering::Relaxed) {
            if tokio::fs::metadata(&job.path)
                .await
                .is_ok_and(|metadata| metadata.is_file())
            {
                self.finish(job, Ok(())).await;
                return;
            }
            self.0.disk.started(&job.path);
            if let Some(tracker) = &job.tracker {
                tracker.started(&job.path);
            }
            if job.draw_progress {
                *job.presentation.lock().unwrap() = Some(DownloadProgress::new(&job.path));
            }
            tracing::info!(file = %job.path.file_name().unwrap_or_default().to_string_lossy(),
                queued_ms = job.queued.elapsed().as_millis() as u64, "resource download started");
        }
        if let Some(tracker) = &job.tracker {
            tracker.activity(&job.path, true);
        }
        let observed = job.clone();
        let disk = self.0.disk.clone();
        let progress: DownloadProgressFn =
            Arc::new(move |increment, total| observed.report(&disk, increment, total));
        let outcome = {
            let mut transfer = job.transfer.lock().await;
            match transfer.advance(self.0.limits, progress).await {
                Ok(true) => transfer.publish().await.map(|()| true),
                other => other,
            }
        };
        match outcome {
            Ok(false) => {
                if let Some(tracker) = &job.tracker {
                    tracker.activity(&job.path, false);
                }
                job.state.lock().unwrap().last_served = Instant::now();
                self.0.queue.lock().unwrap().pending.push_back(job.clone());
            }
            Ok(true) => self.finish(job, Ok(())).await,
            Err(error) => self.finish(job, Err(error)).await,
        }
    }

    async fn finish(&self, job: &Arc<Job>, outcome: Result<(), String>) {
        if outcome.is_err() {
            job.transfer.lock().await.discard().await;
        }
        self.0.disk.finished(&job.path, outcome.is_ok());
        if let Some(tracker) = &job.tracker {
            let bytes = if outcome.is_ok() {
                tokio::fs::metadata(&job.path)
                    .await
                    .ok()
                    .map(|metadata| metadata.len())
            } else {
                None
            };
            tracker.finished(&job.path, bytes);
        }
        if let Some(presentation) = job.presentation.lock().unwrap().take() {
            presentation.finish(outcome.is_ok());
        }
        let mut queue = self.0.queue.lock().unwrap();
        queue.jobs.remove(&job.path);
        job.done
            .send_replace(Some(outcome.map_err(Arc::<str>::from)));
    }
}
