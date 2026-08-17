/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! OpenAI-compatible API surface: every handler here speaks the OpenAI wire
//! format (including the error envelope in `handlers::error`) so off-the-shelf
//! OpenAI clients work against muna-server unchanged.

mod completions;
mod embeddings;
mod images;
mod models;

pub(super) use completions::chat_completions;
pub(super) use embeddings::embeddings;
pub(super) use images::image_generations;
pub(super) use models::models;
