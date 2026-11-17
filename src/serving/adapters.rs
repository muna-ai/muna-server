/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Lazy protocol adapters for one immutable predictor signature.

use std::sync::{Arc, OnceLock};

use muna::beta::adapters::{
    AnthropicMessagesAdapter, OpenAICompletionsAdapter, OpenAIDepthAdapter,
    OpenAIEmbeddingsAdapter, OpenAIImageEditAdapter, OpenAIImageGenerationAdapter,
    OpenAITranscriptionsAdapter,
};
use muna::client::{MunaError, Result};
use muna::Signature;

type Slot<T> = OnceLock<std::result::Result<Arc<T>, String>>;

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

    pub fn transcriptions(&self) -> Result<Arc<OpenAITranscriptionsAdapter>> {
        cached(&self.transcriptions, || {
            OpenAITranscriptionsAdapter::from_signature(&self.signature)
        })
    }
}

fn cached<T>(slot: &Slot<T>, create: impl FnOnce() -> Result<T>) -> Result<Arc<T>> {
    slot.get_or_init(|| {
        create().map(Arc::new).map_err(|error| match error {
            MunaError::InvalidInput(message) => message,
            error => error.to_string(),
        })
    })
    .as_ref()
    .map(Arc::clone)
    .map_err(|message| MunaError::InvalidInput(message.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
                            Ok(42)
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
    fn unsupported_surfaces_are_cached_without_changing_the_error() {
        let slot = Slot::<usize>::new();
        let mut calls = 0;
        for _ in 0..2 {
            let error = cached(&slot, || {
                calls += 1;
                Err(MunaError::InvalidInput("unsupported signature".into()))
            })
            .unwrap_err();
            assert!(
                matches!(error, MunaError::InvalidInput(message) if message == "unsupported signature")
            );
        }
        assert_eq!(calls, 1);
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
