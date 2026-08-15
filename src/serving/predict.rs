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

use std::future::Future;

use futures_util::StreamExt;
use muna::MunaError;

/// Run a muna operation to completion on the blocking pool.
///
/// `Handle::block_on` is legal on blocking-pool threads and panics on core
/// runtime workers -- which is exactly the misuse it should catch.
pub(crate) async fn run<T, F, Fut>(op: F) -> Result<T, MunaError>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, MunaError>>,
    T: Send + 'static,
{
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(op()))
        .await
        .unwrap_or_else(|e| Err(MunaError::Native(format!("prediction task panicked: {e}"))))
}

/// Pump a muna stream into a channel from one blocking thread.
///
/// The stream is created and consumed entirely on the blocking thread; only
/// the items cross threads, so the muna stream type itself never needs to be
/// `Send`. Dropping the receiver ends the pump on its next send (client
/// disconnect).
pub(crate) fn stream<T, F, Fut, S>(op: F) -> tokio::sync::mpsc::Receiver<Result<T, MunaError>>
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
        })
    });
    rx
}
