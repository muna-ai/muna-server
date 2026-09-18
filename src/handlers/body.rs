/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

use serde::Deserialize;

/// A chat request body: muna-rs's request params plus the transport-level
/// `stream` flag that muna-rs deliberately ignores.
///
/// Handlers deserialize straight into muna-rs's `ChatCompletionCreateParams`
/// / `MessageCreateParams` rather than a hand-copied local struct, so a
/// field added in muna-rs is forwarded here (and hashed identically by the
/// control plane's router sidecar) with no server change.
#[derive(Deserialize)]
pub(crate) struct ChatBody<T> {
    /// Whether to stream the response as server-sent events.
    #[serde(default)]
    pub stream: bool,
    /// The muna-rs request params.
    #[serde(flatten)]
    pub params: T,
}

#[cfg(test)]
mod tests {
    use super::*;
    use muna::beta::anthropic::{MessageCreateParams, ThinkingConfig};
    use muna::beta::openai::{ChatCompletionCreateParams, ChatCompletionReasoningEffort};

    #[test]
    fn openai_body_forwards_sampling_and_reasoning_knobs() {
        let body: ChatBody<ChatCompletionCreateParams> = serde_json::from_value(serde_json::json!({
            "model": "@a/x",
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "reasoning_effort": "high",
            "max_completion_tokens": 32,
        })).unwrap();
        assert!(body.stream);
        assert_eq!(body.params.model, "@a/x");
        assert_eq!(body.params.messages.len(), 1);
        assert_eq!(body.params.temperature, Some(0.2));
        assert_eq!(body.params.top_p, Some(0.9));
        assert_eq!(body.params.seed, Some(7));
        assert_eq!(body.params.reasoning_effort, Some(ChatCompletionReasoningEffort::High));
        assert_eq!(body.params.max_completion_tokens, Some(32));
        // `acceleration` is not a wire field; the handler sets it post-parse.
        assert!(body.params.acceleration.is_none());
    }

    #[test]
    fn max_tokens_alias_survives_flatten() {
        // serde's `flatten` filters the buffered map by the inner struct's
        // generated `FIELDS`, which must include aliases for the deprecated
        // OpenAI spelling to keep working through the wrapper.
        let body: ChatBody<ChatCompletionCreateParams> = serde_json::from_value(serde_json::json!({
            "model": "@a/x",
            "messages": [],
            "max_tokens": 64,
        })).unwrap();
        assert!(!body.stream);
        assert_eq!(body.params.max_completion_tokens, Some(64));
    }

    #[test]
    fn anthropic_body_forwards_thinking_and_requires_max_tokens() {
        let body: ChatBody<MessageCreateParams> = serde_json::from_value(serde_json::json!({
            "model": "@a/x",
            "max_tokens": 1024,
            "messages": [{ "role": "user", "content": "hi" }],
            "stream": true,
            "top_k": 40,
            "thinking": { "type": "enabled", "budget_tokens": 2048 },
        })).unwrap();
        assert!(body.stream);
        assert_eq!(body.params.max_tokens, 1024);
        assert_eq!(body.params.top_k, Some(40));
        assert!(matches!(body.params.thinking, Some(ThinkingConfig::Enabled { budget_tokens: 2048, .. })));
        // Missing `max_tokens` is still a deserialization error (400), as before.
        let missing = serde_json::from_value::<ChatBody<MessageCreateParams>>(serde_json::json!({
            "model": "@a/x",
            "messages": [{ "role": "user", "content": "hi" }],
        }));
        assert!(missing.is_err());
    }
}
