/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

//! Deliberate failures in the real server, excluded from default/release builds.
//! Never enable the `crash-reporting-tests` feature on a deployed server.

use crate::state::AppState;
use axum::{extract::Path, http::StatusCode, routing::post, Router};
use std::sync::Arc;

pub fn router() -> Router<Arc<AppState>> {
    Router::new().route("/__test/crash/{mode}", post(trigger))
}

#[inline(never)]
fn server_panic_probe() {
    panic!("deliberate server panic for crash integration test");
}

#[inline(never)]
fn server_signal_probe() {
    unsafe {
        libc::raise(libc::SIGSEGV);
    }
}

async fn trigger(Path(mode): Path<String>) -> StatusCode {
    match mode.as_str() {
        "panic" => server_panic_probe(),
        "blocking-panic" => {
            let _ = tokio::task::spawn_blocking(server_panic_probe).await;
        }
        "segv" => server_signal_probe(),
        _ => return StatusCode::NOT_FOUND,
    }
    StatusCode::NO_CONTENT
}
