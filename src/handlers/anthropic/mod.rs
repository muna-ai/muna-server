/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Anthropic-compatible API surface: handlers here speak the Anthropic wire
//! format (including the `{"type": "error", ...}` envelope in
//! `handlers::error`) so off-the-shelf Anthropic clients work against
//! muna-server unchanged. The `x-api-key` and `anthropic-version` headers are
//! ignored, consistent with the OpenAI routes ignoring `Authorization`.

mod messages;

pub(super) use messages::messages;
