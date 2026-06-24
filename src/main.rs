/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use clap::{Parser, Subcommand};
use muna::types::Acceleration;
use muna::Muna;

mod handlers;

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

pub(crate) struct AppState {
    /// Muna client.
    muna: Muna,
    /// Muna doesn't expose loaded model state, so track successful loads here.
    loaded_models: tokio::sync::RwLock<BTreeMap<String, u64>>,
    /// Serializes predictions: the native libFunction/llama.cpp predictor is not
    /// safe to run concurrently (concurrent predictions abort the process), so
    /// only one chat completion runs at a time.
    predict_lock: Arc<tokio::sync::Mutex<()>>,
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
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(cli.port).await,
        Command::Preload { tags } => preload(tags).await.map_err(|e| e.to_string()),
    }
}

async fn serve(port: u16) -> Result<(), String> {
    // access_key=None -> muna falls back to $MUNA_ACCESS_KEY.
    let state = Arc::new(AppState {
        muna: Muna::new(None, None),
        loaded_models: tokio::sync::RwLock::new(BTreeMap::new()),
        predict_lock: Arc::new(tokio::sync::Mutex::new(())),
    });
    let app = Router::new()
        .route("/", get(handlers::health))
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .fallback(handlers::not_found)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("failed to bind {addr}: {e}"))?;
    tracing::info!("muna-server listening on {addr} (chat -> requested model, local_gpu)");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))
}

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
