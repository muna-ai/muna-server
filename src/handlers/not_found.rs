/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::json;

pub(crate) async fn not_found() -> impl IntoResponse {
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
