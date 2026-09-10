/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Blocking prediction executor.
//!
//! muna-rs keeps inline-FFI semantics: once a predictor is cached and linked,
//! its futures execute the native call synchronously on the polling thread.
//! These helpers offload that work onto tokio's blocking pool so core runtime
//! workers never stall. Both take a delegate that creates the muna future
//! *inside* the blocking thread, so the future never needs to be `Send` and
//! one helper serves every muna operation (raw predictions, the OpenAI
//! client, `images.generate`, the warmup sentinel).
//!
//! Both also take the model's dispatch guard and carry it onto the blocking
//! thread. A compiled predictor that is not continuous must never be
//! invoked concurrently, and a blocking task cannot be cancelled: when a
//! client disconnects, the handler future is dropped but the native call
//! runs to completion. Had the guard lived in the handler, that drop would
//! have released it mid-invocation and admitted the next request into the
//! predictor (the FLUX.2 [klein] crash of Sep 2026). Owning the guard here
//! ties its release to the native call actually returning.

use std::future::Future;

use futures_util::StreamExt;
use muna::MunaError;

/// A model's sequential dispatch guard, or `None` for continuous models.
///
/// Acquired from `Dispatcher::acquire` and handed to [`run`] / [`stream`],
/// which release it only once the native invocation has finished.
pub(crate) type Guard = Option<tokio::sync::OwnedMutexGuard<()>>;

/// Run a muna operation to completion on the blocking pool.
///
/// `guard` is released when the native call returns, even if the caller is
/// dropped first. `Handle::block_on` is legal on blocking-pool threads and
/// panics on core runtime workers -- which is exactly the misuse it should
/// catch.
pub(crate) async fn run<T, F, Fut>(
    guard: Guard,
    op: F
) -> Result<T, MunaError>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, MunaError>>,
    T: Send + 'static,
{
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let result = handle.block_on(op());
        // Named so it outlives the invocation above and drops here, on the
        // blocking thread, after the predictor has returned.
        drop(guard);
        result
    })
        .await
        .unwrap_or_else(|e| Err(MunaError::Native(format!("prediction task panicked: {e}"))))
}

/// Pump a muna stream into a channel from one blocking thread.
///
/// The stream is created and consumed entirely on the blocking thread; only
/// the items cross threads, so the muna stream type itself never needs to be
/// `Send`. Dropping the receiver ends the pump on its next send (client
/// disconnect). `guard` is released after the native stream has been
/// dropped, so a disconnected client cannot admit the next request into a
/// sequential predictor that is still producing.
pub(crate) fn stream<T, F, Fut, S>(
    guard: Guard,
    op: F
) -> tokio::sync::mpsc::Receiver<Result<T, MunaError>>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<S, MunaError>>,
    S: futures_util::Stream<Item = Result<T, MunaError>>,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        handle.block_on(async move {
            let stream = match op().await {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };
            let mut stream = std::pin::pin!(stream);
            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });
        // The native stream was dropped inside `block_on`; only now is the
        // predictor free for the next caller.
        drop(guard);
    });
    rx
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;

    /// How long the fake native call blocks its thread.
    const INVOCATION: Duration = Duration::from_millis(150);

    async fn wait_until(flag: &AtomicBool) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while !flag.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("flag never set");
    }

    /// The Sep 2026 FLUX crash: the caller cancels mid-invocation, the
    /// blocking native call keeps running, and the guard must keep the
    /// predictor closed until that call returns -- not until the caller's
    /// future is dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_releases_guard_when_invocation_returns_not_when_caller_drops() {
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        let guard = lock.clone().lock_owned().await;
        let started = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let (op_started, op_finished) = (started.clone(), finished.clone());
        let caller = run(Some(guard), move || async move {
            op_started.store(true, Ordering::SeqCst);
            // Blocking sleep: this is the native predictor, which cannot be
            // interrupted once entered.
            std::thread::sleep(INVOCATION);
            op_finished.store(true, Ordering::SeqCst);
            Ok::<(), MunaError>(())
        });
        // Cancel the caller once the invocation is under way.
        let cancelled = tokio::time::timeout(Duration::from_millis(30), caller).await;
        assert!(cancelled.is_err(), "caller should have been cancelled");
        wait_until(&started).await;
        assert!(!finished.load(Ordering::SeqCst));
        // The predictor is still running: nobody else may enter.
        assert!(lock.try_lock().is_err(), "guard released while invocation in flight");
        wait_until(&finished).await;
        let _reacquired = tokio::time::timeout(Duration::from_secs(1), lock.lock())
            .await
            .expect("guard must be released once the invocation returns");
    }

    /// Same contract for streams: a disconnected client drops the receiver,
    /// but the guard is held until the native stream itself is dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stream_releases_guard_after_native_stream_drops() {
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        let guard = lock.clone().lock_owned().await;
        let stream_dropped = Arc::new(AtomicBool::new(false));
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let flag = stream_dropped.clone();
        let mut rx = stream(Some(guard), move || async move {
            let owner = DropFlag(flag);
            // Each item costs a blocking sleep, like a token from the engine.
            let items = futures_util::stream::iter(0..3).map(move |i| {
                let _keep = &owner;
                std::thread::sleep(Duration::from_millis(40));
                Ok::<i32, MunaError>(i)
            });
            Ok(items)
        });
        // Take one item, then disconnect.
        let first = rx.recv().await.expect("first item").expect("ok item");
        assert_eq!(first, 0);
        assert!(lock.try_lock().is_err(), "guard released before the stream ended");
        drop(rx);
        // The pump notices on its next send, drops the native stream, and
        // only then releases the guard.
        wait_until(&stream_dropped).await;
        let _reacquired = tokio::time::timeout(Duration::from_secs(1), lock.lock())
            .await
            .expect("guard must be released once the native stream is dropped");
    }
}
