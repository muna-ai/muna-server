/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::future::IntoFuture;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use tokio::signal::unix::{signal, SignalKind};
use tokio_util::task::TaskTracker;

mod client;
mod control;
mod crash_reporting;
#[cfg(feature = "crash-reporting-tests")]
mod crash_reporting_tests;
mod handlers;
mod notifications;
mod platform;
mod serving;
mod state;

use state::{AppState, NodeContext, ServerMode};

/// How long in-flight requests get to finish after the first signal.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// How long background loops get to exit after the server stops; bounds
/// the heartbeat's final report.
const LOOP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Parser)]
#[command(
    name = "muna-server",
    version,
    about = "Muna prediction server."
)]
struct Cli {
    /// Port the HTTP server listens on.
    #[arg(long, default_value = "8000", env = "PORT")]
    port: u16,
    /// Control plane base URL. Enables the heartbeat loop and KV event relay.
    #[arg(long, env = "MUNA_SERVER_CONTROL_PLANE_URL")]
    control_plane_url: Option<String>,
    /// Node identity assigned by the control plane at provision time.
    #[arg(long, env = "MUNA_SERVER_ID")]
    node_id: Option<String>,
    /// Heartbeat cadence in seconds.
    #[arg(long, default_value = "5", env = "MUNA_SERVER_HEARTBEAT_INTERVAL")]
    heartbeat_interval: u64,
    /// KV relay flush cadence in seconds.
    #[arg(long, default_value = "1", env = "MUNA_SERVER_KV_FLUSH_INTERVAL")]
    kv_flush_interval: u64,
    /// Predictor tags this server serves (comma-separated). When set, the
    /// models are loaded eagerly at boot and requests for any other tag are
    /// rejected with 404. Unset leaves any tag loadable on demand.
    #[arg(long, env = "MUNA_SERVER_MODELS", value_delimiter = ',')]
    models: Vec<String>,
}

fn main() {
    crash_reporting::run_reporter_if_requested();
    crash_reporting::install();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .on_thread_start(crash_reporting::register_thread)
        .build()
        .expect("could not build Tokio runtime")
        .block_on(async_main());
}

async fn async_main() {
    let _ = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "muna_server=info".into()),
        )
        .init();

    if let Err(e) = serve(&Cli::parse()).await {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

async fn serve(cli: &Cli) -> Result<(), String> {
    let mode = match (&cli.control_plane_url, &cli.node_id) {
        (Some(url), Some(node_id)) => ServerMode::Managed(NodeContext {
            node_id: node_id.clone(),
            control_plane_url: url.clone(),
            heartbeat_interval: Duration::from_secs(cli.heartbeat_interval.max(1)),
            kv_flush_interval: Duration::from_secs(cli.kv_flush_interval.max(1)),
        }),
        (Some(_), None) => {
            return Err("--control-plane-url requires --node-id".into());
        }
        _ => ServerMode::Standalone,
    };
    // There is no process-wide Muna client: the registry builds one keyed
    // ServerClient-backed instance per loaded model (see ReadyModel::muna).
    let pinned: Option<std::collections::HashSet<String>> = if cli.models.is_empty() {
        None
    } else {
        Some(cli.models.iter().cloned().collect())
    };
    let state = Arc::new(AppState::new(pinned, mode)?);
    // Warm the platform driver off the runtime thread: a cold driver load
    // under disk pressure can take seconds, and the listener must bind
    // regardless. Fail-soft: the engine's own init still works without it.
    {
        let platform = state.platform.clone();
        tracing::info!(vendor = ?platform.vendor(), "platform detected");
        tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let outcome = platform.warm();
            let elapsed_ms = started.elapsed().as_millis();
            match outcome {
                Ok(Some(platform::WarmReport { devices })) => tracing::info!(
                    devices,
                    elapsed_ms,
                    "platform driver warm complete"
                ),
                Ok(None) => tracing::info!(
                    elapsed_ms,
                    "platform driver had nothing to warm"
                ),
                Err(error) => tracing::warn!(
                    %error,
                    elapsed_ms,
                    "platform driver warm failed; continuing"
                ),
            }
        });
    }
    // Eager load: fire-and-forget warms so the port binds immediately;
    // mid-load requests get 429 + Retry-After.
    for tag in cli.models.iter().filter(|_| matches!(&state.mode, ServerMode::Standalone)) {
        tracing::info!(tag = %tag, "eager-loading pinned model");
        state.registry.warm(tag);
    }
    // Background loops exit when `state.shutdown` is cancelled; the tracker
    // lets shutdown wait for them (the heartbeat's final report included).
    let loops = TaskTracker::new();
    if matches!(&state.mode, ServerMode::Managed(_)) {
        loops.spawn(control::heartbeat::run(state.clone()));
        loops.spawn(control::kv_relay::run(state.clone()));
        tracing::info!("control-plane mode: heartbeat + KV relay enabled");
    }
    // Co-resident NanoSGL engines attach to the host-owned Corby runtime.
    loops.spawn(serving::lease_weights::run(state.clone()));
    loops.close();
    tokio::spawn(watch_signals(state.clone()));
    let app = {
        let router = handlers::router();
        #[cfg(feature = "crash-reporting-tests")]
        let router = router.merge(crash_reporting_tests::router());
        router.with_state(state.clone())
    };
    // Bind listener
    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind {addr}: {e}"))?;
    tracing::info!("muna-server listening on {addr}");
    // Serve until shutdown, then give in-flight requests (long streams
    // included) a bounded window to finish.
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(state.shutdown.clone().cancelled_owned())
        .into_future();
    let drain_deadline = async {
        state.shutdown.cancelled().await;
        tokio::time::sleep(DRAIN_TIMEOUT).await;
    };
    tokio::select! {
        result = server => result.map_err(|e| format!("server error: {e}"))?,
        _ = drain_deadline => tracing::warn!(
            timeout_s = DRAIN_TIMEOUT.as_secs(),
            "in-flight requests did not finish in time; closing them"
        ),
    }
    if tokio::time::timeout(LOOP_SHUTDOWN_TIMEOUT, loops.wait()).await.is_err() {
        tracing::warn!("background loops did not exit in time");
    }
    // Withdraw predictor registrations before stopping their coordinator.
    let unload = async {
        for (tag, _) in state.registry.snapshot() {
            state.registry.unload(&tag).await;
        }
    };
    if tokio::time::timeout(DRAIN_TIMEOUT, unload).await.is_err() {
        tracing::warn!("predictor unload exceeded shutdown deadline");
    }
    if let Err(error) = state.lease.shutdown() {
        // Busy runtimes stay pinned; never revoke a lease from native work.
        tracing::warn!(%error, "Corby runtime retained while engines drain");
    }
    Ok(())
}

/// First SIGINT or SIGTERM: drain and cancel `state.shutdown`. Second:
/// exit immediately, for when a native call or stream holds up shutdown.
async fn watch_signals(state: Arc<AppState>) {
    let mut terminate = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
    let mut interrupt = signal(SignalKind::interrupt()).expect("failed to install SIGINT handler");
    tokio::select! {
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
    tracing::info!("shutting down");
    // Reject new inference requests on connections still open, and report
    // `draining: true` in the final heartbeat.
    state.set_draining(true);
    state.shutdown.cancel();
    tokio::select! {
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
    tracing::warn!("second signal; exiting immediately");
    std::process::exit(130);
}
