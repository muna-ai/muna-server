/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Lazy protocol adapters for one immutable predictor signature.

use std::sync::{Arc, OnceLock};

use muna::beta::adapters::{
    AnthropicMessagesAdapter, OpenAICompletionsAdapter, OpenAIDepthAdapter,
    OpenAIEmbeddingsAdapter, OpenAIImageEditAdapter, OpenAIImageGenerationAdapter,
    OpenAIModel3DAdapter, OpenAITranscriptionsAdapter,
};
use muna::client::{MunaError, Result};
use muna::Signature;

type Slot<T> = OnceLock<Option<Arc<T>>>;

/// A signature belongs to this set for its entire lifetime. Cache failures too:
/// an unsupported surface cannot become supported until the signature changes.
pub(crate) struct Adapters {
    pub signature: Signature,
    completions: Slot<OpenAICompletionsAdapter>,
    messages: Slot<AnthropicMessagesAdapter>,
    embeddings: Slot<OpenAIEmbeddingsAdapter>,
    images: Slot<OpenAIImageGenerationAdapter>,
    edits: Slot<OpenAIImageEditAdapter>,
    depth: Slot<OpenAIDepthAdapter>,
    model3d: Slot<OpenAIModel3DAdapter>,
    transcriptions: Slot<OpenAITranscriptionsAdapter>,
}

impl Adapters {
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            completions: OnceLock::new(),
            messages: OnceLock::new(),
            embeddings: OnceLock::new(),
            images: OnceLock::new(),
            edits: OnceLock::new(),
            depth: OnceLock::new(),
            model3d: OnceLock::new(),
            transcriptions: OnceLock::new(),
        }
    }

    pub fn completions(&self) -> Result<Arc<OpenAICompletionsAdapter>> {
        cached(&self.completions, || {
            OpenAICompletionsAdapter::from_signature(&self.signature)
        })
    }

    pub fn messages(&self) -> Result<Arc<AnthropicMessagesAdapter>> {
        cached(&self.messages, || {
            AnthropicMessagesAdapter::from_signature(&self.signature)
        })
    }

    pub fn embeddings(&self) -> Result<Arc<OpenAIEmbeddingsAdapter>> {
        cached(&self.embeddings, || {
            OpenAIEmbeddingsAdapter::from_signature(&self.signature)
        })
    }

    pub fn images(&self) -> Result<Arc<OpenAIImageGenerationAdapter>> {
        cached(&self.images, || {
            OpenAIImageGenerationAdapter::from_signature(&self.signature)
        })
    }

    pub fn edits(&self) -> Result<Arc<OpenAIImageEditAdapter>> {
        cached(&self.edits, || {
            OpenAIImageEditAdapter::from_signature(&self.signature)
        })
    }

    pub fn depth(&self) -> Result<Arc<OpenAIDepthAdapter>> {
        cached(&self.depth, || {
            OpenAIDepthAdapter::from_signature(&self.signature)
        })
    }

    pub fn model3d(&self) -> Result<Arc<OpenAIModel3DAdapter>> {
        cached(&self.model3d, || {
            OpenAIModel3DAdapter::from_signature(&self.signature)
        })
    }

    pub fn transcriptions(&self) -> Result<Arc<OpenAITranscriptionsAdapter>> {
        cached(&self.transcriptions, || {
            OpenAITranscriptionsAdapter::from_signature(&self.signature)
        })
    }
}

fn cached<T>(slot: &Slot<T>, create: impl FnOnce() -> Option<T>) -> Result<Arc<T>> {
    slot.get_or_init(|| create().map(Arc::new))
        .as_ref()
        .map(Arc::clone)
        .ok_or_else(|| {
            MunaError::InvalidInput("Predictor signature is incompatible with this endpoint".into())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn model3d_adapter_is_cached_for_the_signature() {
        let adapters = Adapters::new(serde_json::from_value(serde_json::json!({
            "inputs":[
                {"name":"images","dtype":"image_list","denotation":"openai.chat.completions.images"},
                {"name":"prompt","dtype":"string","denotation":"openai.images.prompt","optional":true}
            ],
            "outputs":[{"name":"asset","dtype":"binary","denotation":"3d"}]
        })).unwrap());
        let first = adapters.model3d().unwrap();
        assert!(Arc::ptr_eq(&first, &adapters.model3d().unwrap()));
        assert!(adapters.edits().is_err());
        assert!(adapters.depth().is_err());
    }

    #[test]
    fn concurrent_requests_initialize_once_and_share_the_adapter() {
        let slot = Slot::<usize>::new();
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let tasks: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        cached(&slot, || {
                            calls.fetch_add(1, Ordering::SeqCst);
                            Some(42)
                        })
                        .unwrap()
                    })
                })
                .collect();
            let adapters: Vec<_> = tasks.into_iter().map(|task| task.join().unwrap()).collect();
            assert!(adapters
                .iter()
                .all(|adapter| Arc::ptr_eq(adapter, &adapters[0])));
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unsupported_surfaces_are_cached_as_invalid_input() {
        let slot = Slot::<usize>::new();
        let mut calls = 0;
        for _ in 0..2 {
            let error = cached(&slot, || {
                calls += 1;
                None
            })
            .unwrap_err();
            assert!(
                matches!(error, MunaError::InvalidInput(message) if message == "Predictor signature is incompatible with this endpoint")
            );
        }
        assert_eq!(calls, 1);
    }
    #[test]
    fn chat_surfaces_require_explicit_messages_denotation() {
        let mut signature: Signature = serde_json::from_value(serde_json::json!({
            "inputs":[{"name":"conversation", "dtype":"list"}],
            "outputs":[{"name":"chunk", "dtype":"dict", "schema":{"title":"ChatCompletionChunk"}}],
        }))
        .unwrap();
        let unannotated = Adapters::new(signature.clone());
        assert!(matches!(
            unannotated.completions(),
            Err(MunaError::InvalidInput(_))
        ));
        assert!(matches!(
            unannotated.messages(),
            Err(MunaError::InvalidInput(_))
        ));

        signature.inputs[0].denotation = Some("openai.chat.completions.messages".into());
        let annotated = Adapters::new(signature);
        assert_eq!(
            annotated.completions().unwrap().messages.name,
            "conversation"
        );
        assert_eq!(annotated.messages().unwrap().messages.name, "conversation");
    }

    fn signature(input: &str) -> Signature {
        serde_json::from_value(serde_json::json!({
            "inputs":[{"name":input, "dtype":"list"}],
            "outputs":[{"name":"vectors", "dtype":"float32", "denotation":"embedding"}],
        }))
        .unwrap()
    }

    #[test]
    fn model_instances_keep_independent_adapters_for_their_own_signature() {
        let old = Adapters::new(signature("old_texts"));
        let new = Adapters::new(signature("new_texts"));
        let first = old.embeddings().unwrap();
        assert!(Arc::ptr_eq(&first, &old.embeddings().unwrap()));
        assert!(!Arc::ptr_eq(&first, &new.embeddings().unwrap()));
        assert_eq!(first.input.name, "old_texts");
        assert_eq!(new.embeddings().unwrap().input.name, "new_texts");
    }
}
