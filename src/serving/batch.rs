/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Signature-driven batching plan and batch-key fingerprints.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use muna::types::{BatchMode, Signature, Value};

/// How the server dispatches predictions for a model, derived from its
/// signature at load time.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BatchPlan {
    /// No batch config on any input: one prediction at a time (with a per-model mutex).
    Sequential,
    /// Static or dynamic batching: buffer into a queue, merge up to capacity,
    /// flush at capacity or deadline. The server never pads a partial batch --
    /// it cannot know a valid pad value for opaque inputs. A predictor
    /// compiled with a rigid batch shape must pad internally,
    /// so both modes dispatch identically here.
    Buffered {
        /// Input parameters that are merged across requests.
        params: HashSet<String>,
        /// Maximum total item count across merged requests.
        capacity: usize,
    },
    /// Continuous batching: submit concurrently, the compiled engine batches
    /// internally. No lock, ever.
    Continuous,
}

impl BatchPlan {

    /// Derive the dispatch plan from a predictor signature.
    pub(crate) fn from_signature(signature: &Signature) -> Self {
        let batched: Vec<_> = signature.inputs.iter()
            .filter(|p| p.batch.is_some())
            .collect();
        if batched.is_empty() {
            return Self::Sequential;
        }
        // A continuous parameter means the engine owns synchronization for
        // the whole predictor; buffering would only add latency.
        if batched.iter().any(|p| {
            p.batch.as_ref().is_some_and(|b| b.mode == BatchMode::Continuous)
        }) {
            return Self::Continuous;
        }
        let params: HashSet<String> = batched.iter().map(|p| p.name.clone()).collect();
        let capacity = batched.iter()
            .filter_map(|p| p.batch.as_ref().and_then(|b| b.capacity))
            .min()
            .unwrap_or(1);
        Self::Buffered { params, capacity }
    }
}

/// Compute a deterministic key from broadcast (i.e. non-batch) parameter values.
///
/// Two requests can only be merged into the same batch if their batch keys
/// are identical, ensuring broadcast parameters are never silently
/// overwritten.
pub(crate) fn compute_batch_key(
    inputs: &HashMap<String, Value>,
    batch_params: &HashSet<String>,
) -> String {
    let mut parts: Vec<String> = inputs.iter()
        .filter(|(name, _)| !batch_params.contains(name.as_str()))
        .map(|(name, value)| format!("{}={}", name, value_fingerprint(value)))
        .collect();
    parts.sort();
    parts.join("&")
}

/// Number of batchable items a request carries: the maximum list length
/// among its batch parameters (a non-list batch param counts as one item).
pub(crate) fn item_count(
    inputs: &HashMap<String, Value>,
    batch_params: &HashSet<String>,
) -> usize {
    batch_params.iter()
        .filter_map(|name| inputs.get(name))
        .map(|value| match value {
            Value::List(items) => items.len(),
            _ => 1,
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

/// Produce a deterministic string for a single value, used for batch key
/// comparison.
///
/// Floats use exact bit representation (`to_bits`) rather than epsilon
/// comparison because these values come directly from user JSON input, not
/// from arithmetic. Opaque types (tensors, images, binary) get a unique
/// monotonic ID so they never match, preventing batching on values we can't
/// meaningfully compare.
pub(crate) fn value_fingerprint(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::String(s) => format!("s:{s}"),
        Value::Float(f) => format!("f:{}", f.to_bits()),
        Value::Double(d) => format!("d:{}", d.to_bits()),
        Value::Int(i) => format!("i:{i}"),
        Value::Long(l) => format!("l:{l}"),
        Value::Bool(b) => format!("b:{b}"),
        Value::List(l) => format!("list:{}", serde_json::to_string(l).unwrap_or_default()),
        Value::Dict(d) => format!("dict:{}", serde_json::to_string(d).unwrap_or_default()),
        _ => {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            format!("__opaque_{}__", COUNTER.fetch_add(1, Ordering::Relaxed))
        }
    }
}

#[cfg(test)]
mod tests {
    use muna::types::{BatchConfig, Dtype, Parameter, Tensor, TensorData};

    use super::*;

    fn param(name: &str, dtype: Dtype, batch: Option<BatchConfig>) -> Parameter {
        Parameter {
            name: name.into(),
            dtype: Some(dtype),
            description: None,
            denotation: None,
            optional: None,
            enumeration: None,
            schema: None,
            min: None,
            max: None,
            sample_rate: None,
            batch,
        }
    }

    fn signature(inputs: Vec<Parameter>) -> Signature {
        Signature { inputs, outputs: vec![] }
    }

    #[test]
    fn plan_no_batch_params_is_sequential() {
        let sig = signature(vec![
            param("texts", Dtype::List, None),
            param("dimensions", Dtype::Int32, None),
        ]);
        assert_eq!(BatchPlan::from_signature(&sig), BatchPlan::Sequential);
    }

    #[test]
    fn plan_continuous_param_wins() {
        let sig = signature(vec![
            param("messages", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Continuous,
                capacity: None,
            })),
            param("images", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Dynamic,
                capacity: Some(8),
            })),
        ]);
        assert_eq!(BatchPlan::from_signature(&sig), BatchPlan::Continuous);
    }

    #[test]
    fn plan_dynamic_uses_min_capacity() {
        let sig = signature(vec![
            param("texts", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Dynamic,
                capacity: Some(128),
            })),
            param("images", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Dynamic,
                capacity: Some(32),
            })),
            param("dimensions", Dtype::Int32, None),
        ]);
        match BatchPlan::from_signature(&sig) {
            BatchPlan::Buffered { params, capacity } => {
                assert_eq!(params, HashSet::from(["texts".into(), "images".into()]));
                assert_eq!(capacity, 32);
            }
            other => panic!("expected Buffered, got {other:?}"),
        }
    }

    #[test]
    fn plan_static_and_dynamic_derive_identical_plans() {
        // Static and dynamic differ only in the compiled predictor's shape
        // contract (a static predictor pads internally); the server
        // dispatches both identically.
        let static_sig = signature(vec![
            param("texts", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Static,
                capacity: Some(4),
            })),
        ]);
        let dynamic_sig = signature(vec![
            param("texts", Dtype::List, Some(BatchConfig {
                mode: BatchMode::Dynamic,
                capacity: Some(4),
            })),
        ]);
        let static_plan = BatchPlan::from_signature(&static_sig);
        assert_eq!(static_plan, BatchPlan::from_signature(&dynamic_sig));
        match static_plan {
            BatchPlan::Buffered { capacity, .. } => assert_eq!(capacity, 4),
            other => panic!("expected Buffered, got {other:?}"),
        }
    }

    #[test]
    fn batch_key_same_broadcast_values_match() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec![])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec![])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        assert_eq!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
    }

    #[test]
    fn batch_key_different_broadcast_values_differ() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec![])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec![])),
            ("dimensions".into(), Value::Int(20)),
        ]);
        assert_ne!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
    }

    #[test]
    fn batch_key_excludes_batch_params() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec![1.into()])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec![1.into(), 2.into(), 3.into()])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        assert_eq!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
    }

    #[test]
    fn batch_key_missing_optional_broadcast_differs() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec![])),
            ("dimensions".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec![])),
        ]);
        assert_ne!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
    }

    #[test]
    fn batch_key_floats_compare_by_bits() {
        let bp: HashSet<String> = HashSet::new();
        let a = HashMap::from([("t".to_string(), Value::Float(0.5))]);
        let b = HashMap::from([("t".to_string(), Value::Float(0.5))]);
        let c = HashMap::from([("t".to_string(), Value::Float(0.25))]);
        assert_eq!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
        assert_ne!(compute_batch_key(&a, &bp), compute_batch_key(&c, &bp));
    }

    #[test]
    fn batch_key_opaque_values_never_match() {
        let bp: HashSet<String> = HashSet::new();
        let tensor = || Value::Tensor(Tensor {
            data: TensorData::Float32(vec![1.0]),
            shape: vec![1],
        });
        let a = HashMap::from([("t".to_string(), tensor())]);
        let b = HashMap::from([("t".to_string(), tensor())]);
        assert_ne!(compute_batch_key(&a, &bp), compute_batch_key(&b, &bp));
    }

    #[test]
    fn item_count_uses_max_batch_list_len() {
        let bp: HashSet<String> = HashSet::from(["texts".into(), "images".into()]);
        let inputs = HashMap::from([
            ("texts".to_string(), Value::List(vec![1.into(), 2.into(), 3.into()])),
            ("images".to_string(), Value::List(vec![1.into()])),
            ("dims".to_string(), Value::Int(10)),
        ]);
        assert_eq!(item_count(&inputs, &bp), 3);
    }

    #[test]
    fn item_count_defaults_to_one() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let inputs = HashMap::from([("dims".to_string(), Value::Int(10))]);
        assert_eq!(item_count(&inputs, &bp), 1);
    }
}
