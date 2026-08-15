/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use clap::{Parser, Subcommand};
use muna::types::Acceleration;
use muna::Muna;

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
    #[arg(long, default_value = "8000", env = "PORT", global = true)]
    port: u16,
    /// Control plane base URL. Enables the heartbeat loop and KV event relay.
    #[arg(long, env = "MUNA_CONTROL_PLANE_URL", global = true)]
    control_plane_url: Option<String>,
    /// Node identity assigned by the control plane at provision time.
    #[arg(long, env = "MUNA_NODE_ID", global = true)]
    node_id: Option<String>,
    /// Heartbeat cadence in seconds.
    #[arg(long, default_value = "5", env = "MUNA_HEARTBEAT_INTERVAL", global = true)]
    heartbeat_interval: u64,
    /// KV relay flush cadence in seconds.
    #[arg(long, default_value = "1", env = "MUNA_KV_FLUSH_INTERVAL", global = true)]
    kv_flush_interval: u64,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the OpenAI-compatible HTTP server.
    Serve,
    /// Preload one or more predictor tags and exit.
    Preload {
        /// Predictor tags to preload.
        #[arg(required = true)]
        tags: Vec<String>,
    },
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

    if let Err(e) = run().await {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let mut cli = Cli::parse();
    match cli.command.take().unwrap_or(Command::Serve) {
        Command::Serve => serve(&cli).await,
        Command::Preload { tags } => preload(tags).await.map_err(|e| e.to_string()),
    }
}

async fn serve(cli: &Cli) -> Result<(), String> {
    let node = match (&cli.control_plane_url, &cli.node_id) {
        (Some(url), Some(node_id)) => Some(NodeContext {
            node_id: node_id.clone(),
            control_plane_url: url.clone(),
            heartbeat_interval: Duration::from_secs(cli.heartbeat_interval.max(1)),
            kv_flush_interval: Duration::from_secs(cli.kv_flush_interval.max(1)),
            event_callbacks: tokio::sync::watch::channel(Vec::new()).0,
        }),
        (Some(_), None) => {
            return Err("--control-plane-url requires --node-id".into());
        }
        _ => None,
    };
    // access_key=None -> muna falls back to $MUNA_ACCESS_KEY.
    let muna = Arc::new(Muna::new(None, None));
    let state = Arc::new(AppState::new(muna, node));
    if state.node.is_some() {
        tokio::spawn(control::heartbeat::run(state.clone()));
        tokio::spawn(control::kv_relay::run(state.clone()));
        tracing::info!("control-plane mode: heartbeat + KV relay enabled");
    }
    let app = Router::new()
        // Health and management
        .route("/", get(handlers::health))
        .route("/health", get(handlers::health))
        .route("/status", get(handlers::status))
        .route("/drain", post(handlers::drain))
        // Muna remote prediction
        .route("/v1/predictions/remote", post(handlers::predictions))
        // OpenAI compatibility
        .route("/v1/models", get(handlers::models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/images/generations", post(handlers::image_generations))
        // Fallbacks
        .fallback(handlers::not_found)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], cli.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind {addr}: {e}"))?;
    tracing::info!("muna-server listening on {addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

/// Download-only preload (empty inputs): embeds predictor resources into a
/// deploy image at build time without loading the engine.
async fn preload(tags: Vec<String>) -> Result<(), muna::MunaError> {
    let muna = Muna::new(None, None);
    for tag in tags {
        println!("Preloading {tag}");
        let prediction = muna
            .predictions
            .create(
                &tag,
                Some(HashMap::<String, muna::types::Value>::new()),
                Some(Acceleration::LocalGpu),
                None,
                None,
            )
            .await?;
        let resource_count = prediction.resources.as_ref().map_or(0, Vec::len);
        println!("Preloaded {tag} ({resource_count} resources)");
    }
    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    tracing::info!("shutting down");
}
