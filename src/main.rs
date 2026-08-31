/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

mod client;
mod control;
mod handlers;
mod metrics;
mod serving;
mod state;

use state::{AppState, NodeContext};

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

#[tokio::main]
async fn main() {
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
    let node = match (&cli.control_plane_url, &cli.node_id) {
        (Some(url), Some(node_id)) => Some(NodeContext {
            node_id: node_id.clone(),
            control_plane_url: url.clone(),
            heartbeat_interval: Duration::from_secs(cli.heartbeat_interval.max(1)),
            kv_flush_interval: Duration::from_secs(cli.kv_flush_interval.max(1)),
        }),
        (Some(_), None) => {
            return Err("--control-plane-url requires --node-id".into());
        }
        _ => None,
    };
    // There is no process-wide Muna client: the registry builds one keyed
    // ServerClient-backed instance per loaded model (see ReadyModel::muna).
    let pinned: Option<std::collections::HashSet<String>> = if cli.models.is_empty() {
        None
    } else {
        Some(cli.models.iter().cloned().collect())
    };
    let state = Arc::new(AppState::new(pinned, node));
    // Eager load: fire-and-forget warms so the port binds immediately;
    // mid-load requests get 429 + Retry-After.
    for tag in &cli.models {
        tracing::info!(tag = %tag, "eager-loading pinned model");
        state.registry.warm(tag);
    }
    if state.node.is_some() {
        tokio::spawn(control::heartbeat::run(state.clone()));
        tokio::spawn(control::kv_relay::run(state.clone()));
        tracing::info!("control-plane mode: heartbeat + KV relay enabled");
    }
    // Co-resident NanoSGL engines arbitrate GPU time through the device lease
    tokio::spawn(serving::lease::run(state.clone()));
    let app = handlers::router().with_state(state);
    // Bind listener
    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind {addr}: {e}"))?;
    tracing::info!("muna-server listening on {addr}");
    // Serve
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    tracing::info!("shutting down");
}
