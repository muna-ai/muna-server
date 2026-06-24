//! OpenAI-compatible HTTP server backed by a local Muna predictor.
//!
//! `/v1/chat/completions` runs the requested predictor through the `muna` crate
//! with `local_gpu` acceleration, relaying output as OpenAI-style JSON or
//! SSE chunks. `/v1/models` reports the models this process has loaded.
//!
//! muna reads the access key from $MUNA_ACCESS_KEY. It links a native libFunction.so
//! (fetched by its build.rs from cdn.fxn.ai) and runs the model via llama.cpp, so this
//! is a glibc-dynamic binary that needs libFunction.so reachable at runtime
//! (LD_LIBRARY_PATH or an $ORIGIN-adjacent copy).

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Parser, Subcommand};
use futures_util::{stream, StreamExt};
use muna::beta::openai::{ChatCompletionCreateParams, ChatCompletionMessage};
use muna::types::Acceleration;
use muna::Muna;
use serde::Deserialize;
use serde_json::{json, Value};

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

struct AppState {
    /// Muna client.
    muna: Muna,
    /// Muna doesn't expose loaded model state, so track successful loads here.
    loaded_models: tokio::sync::RwLock<BTreeMap<String, u64>>,
    /// Serializes predictions: the native libFunction/llama.cpp predictor is not
    /// safe to run concurrently (concurrent predictions abort the process), so
    /// only one chat completion runs at a time.
    predict_lock: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Deserialize)]
struct ChatCompletionsRequest {
    model: String,
    #[serde(default)]
    messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default, alias = "max_tokens")]
    max_completion_tokens: Option<i32>,
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
        .route("/", get(health))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found)
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

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let loaded_models = state.loaded_models.read().await;
    let data: Vec<Value> = loaded_models
        .iter()
        .map(|(model, created)| {
            json!({
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": "muna",
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": data,
    }))
}

/// Real chat: run the requested model via muna and relay the result.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Result<Response, AppError> {
    let params = ChatCompletionCreateParams {
        model: req.model,
        messages: req.messages,
        acceleration: Some(Acceleration::LocalGpu),
        max_completion_tokens: req.max_completion_tokens,
        ..Default::default()
    };

    if req.stream {
        stream_chat_completion(state, params).await
    } else {
        create_chat_completion(state, params).await
    }
    .map_err(AppError::from)
}

async fn create_chat_completion(
    state: Arc<AppState>,
    params: ChatCompletionCreateParams,
) -> Result<Response, muna::MunaError> {
    let model = params.model.clone();
    let _guard = state.predict_lock.clone().lock_owned().await;
    let completion = state
        .muna
        .beta
        .openai
        .chat
        .completions
        .create(params)
        .await?;
    mark_model_loaded(&state, model).await;
    Ok(Json(completion).into_response())
}

async fn stream_chat_completion(
    state: Arc<AppState>,
    params: ChatCompletionCreateParams,
) -> Result<Response, muna::MunaError> {
    let model = params.model.clone();
    // Held for the whole Muna stream. Dropping the response body releases the guard.
    let guard = state.predict_lock.clone().lock_owned().await;
    let muna_stream = state
        .muna
        .beta
        .openai
        .chat
        .completions
        .stream(params)
        .await?;
    mark_model_loaded(&state, model).await;
    let event_stream = muna_stream
        .map(move |result| {
            let _guard = &guard;
            match result {
                Ok(chunk) => {
                    let json = serde_json::to_string(&chunk).unwrap_or_default();
                    Ok::<Event, Infallible>(Event::default().data(json))
                }
                Err(e) => {
                    tracing::warn!("muna stream error: {e}");
                    let json = serde_json::to_string(&muna_error_value(&e)).unwrap_or_default();
                    Ok(Event::default().data(json))
                }
            }
        })
        .chain(stream::once(async {
            Ok::<Event, Infallible>(Event::default().data("[DONE]"))
        }));

    Ok(Sse::new(event_stream).into_response())
}

async fn mark_model_loaded(state: &AppState, model: String) {
    let mut loaded_models = state.loaded_models.write().await;
    loaded_models.entry(model).or_insert_with(now);
}

async fn not_found() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": {
                "message": "unknown route (muna-server)",
                "type": "not_found",
            }
        })),
    )
}

struct AppError {
    status: StatusCode,
    body: Value,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

impl From<muna::MunaError> for AppError {
    fn from(e: muna::MunaError) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: muna_error_value(&e),
        }
    }
}

fn muna_error_value(e: &muna::MunaError) -> Value {
    json!({
        "error": {
            "message": e.to_string(),
            "type": "muna_error",
        }
    })
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to listen for ctrl+c");
    tracing::info!("shutting down");
}
