/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Input-merge and result-split plumbing for batched invocations: pure data
//! movement over `Value`s, with no dispatch or queueing state.

use std::collections::{HashMap, HashSet};

use muna::types::{Tensor, TensorData, Value};

/// Merge a batch of compatible requests into one input map: batch params are
/// concatenated in request order; broadcast params come from the first
/// request (identical across the batch by batch-key construction).
pub(super) fn merge_inputs(
    batch: &[&HashMap<String, Value>],
    batch_params: &HashSet<String>,
) -> HashMap<String, Value> {
    if batch.len() == 1 {
        return batch[0].clone();
    }
    let mut merged = HashMap::new();
    for (key, value) in batch[0] {
        if !batch_params.contains(key.as_str()) {
            merged.insert(key.clone(), value.clone());
        }
    }
    for name in batch_params {
        let mut combined = Vec::new();
        for inputs in batch {
            if let Some(Value::List(items)) = inputs.get(name) {
                combined.extend(items.iter().cloned());
            }
        }
        merged.insert(name.clone(), Value::List(combined));
    }
    merged
}

/// Split merged prediction results back per request: tensors split on dim 0
/// by item counts, lists by counts, everything else broadcasts.
pub(super) fn split_results(
    results: Vec<Value>,
    counts: &[usize],
) -> Vec<Vec<Value>> {
    let total: usize = counts.iter().sum();
    let n = counts.len();
    let mut splits: Vec<Vec<Value>> = (0..n).map(|_| Vec::new()).collect();
    for value in results {
        match &value {
            Value::Tensor(tensor)
                if !tensor.shape.is_empty() && tensor.shape[0] as usize == total => {
                let inner: usize = tensor.shape[1..].iter()
                    .map(|&s| s as usize)
                    .product::<usize>()
                    .max(1);
                let mut offset = 0;
                for (i, &count) in counts.iter().enumerate() {
                    let start = offset * inner;
                    let end = (offset + count) * inner;
                    let slice_shape = {
                        let mut s = tensor.shape.clone();
                        s[0] = count as i32;
                        s
                    };
                    splits[i].push(Value::Tensor(Tensor {
                        data: slice_tensor_data(&tensor.data, start, end),
                        shape: slice_shape,
                    }));
                    offset += count;
                }
            }
            Value::List(items) if items.len() == total => {
                let mut offset = 0;
                for (i, &count) in counts.iter().enumerate() {
                    splits[i].push(Value::List(items[offset..offset + count].to_vec()));
                    offset += count;
                }
            }
            other => {
                for s in &mut splits {
                    s.push(other.clone());
                }
            }
        }
    }
    splits
}

fn slice_tensor_data(data: &TensorData, start: usize, end: usize) -> TensorData {
    match data {
        TensorData::Float32(v) => TensorData::Float32(v[start..end].to_vec()),
        TensorData::Float64(v) => TensorData::Float64(v[start..end].to_vec()),
        TensorData::Int8(v) => TensorData::Int8(v[start..end].to_vec()),
        TensorData::Int16(v) => TensorData::Int16(v[start..end].to_vec()),
        TensorData::Int32(v) => TensorData::Int32(v[start..end].to_vec()),
        TensorData::Int64(v) => TensorData::Int64(v[start..end].to_vec()),
        TensorData::Uint8(v) => TensorData::Uint8(v[start..end].to_vec()),
        TensorData::Uint16(v) => TensorData::Uint16(v[start..end].to_vec()),
        TensorData::Uint32(v) => TensorData::Uint32(v[start..end].to_vec()),
        TensorData::Uint64(v) => TensorData::Uint64(v[start..end].to_vec()),
        TensorData::Complex64(v) => TensorData::Complex64(v[start..end].to_vec()),
        TensorData::Complex128(v) => TensorData::Complex128(v[start..end].to_vec()),
        TensorData::Bool(v) => TensorData::Bool(v[start..end].to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_single_item_returns_inputs() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let inputs = HashMap::from([
            ("texts".into(), Value::List(vec!["hello".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let merged = merge_inputs(&[&inputs], &bp);
        assert_eq!(merged.len(), 2);
        assert!(matches!(merged.get("dims"), Some(Value::Int(10))));
    }

    #[test]
    fn merge_concatenates_batch_params() {
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".into(), Value::List(vec!["a".into(), "b".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let b = HashMap::from([
            ("texts".into(), Value::List(vec!["c".into()])),
            ("dims".into(), Value::Int(10)),
        ]);
        let merged = merge_inputs(&[&a, &b], &bp);
        match merged.get("texts") {
            Some(Value::List(items)) => assert_eq!(items.len(), 3),
            other => panic!("expected list of 3, got {other:?}"),
        }
        assert!(matches!(merged.get("dims"), Some(Value::Int(10))));
    }

    #[test]
    fn split_tensor_along_dim0() {
        let tensor = Value::Tensor(Tensor {
            data: TensorData::Float32(vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ]),
            shape: vec![3, 4],
        });
        let splits = split_results(vec![tensor], &[2, 1]);
        assert_eq!(splits.len(), 2);
        match &splits[0][0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 4]);
                assert!(matches!(
                    &t.data,
                    TensorData::Float32(v) if v == &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                ));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
        match &splits[1][0] {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(matches!(
                    &t.data,
                    TensorData::Float32(v) if v == &[9.0, 10.0, 11.0, 12.0]
                ));
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn split_list_by_counts() {
        let list = Value::List(vec![
            "a".into(), "b".into(), "c".into(), "d".into(), "e".into()
        ]);
        let splits = split_results(vec![list], &[3, 2]);
        assert_eq!(splits.len(), 2);
        match &splits[0][0] {
            Value::List(items) => assert_eq!(items.len(), 3),
            other => panic!("expected list, got {other:?}"),
        }
        match &splits[1][0] {
            Value::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn split_scalar_broadcasts() {
        let splits = split_results(vec![Value::Float(42.0)], &[2, 3]);
        assert!(matches!(&splits[0][0], Value::Float(f) if *f == 42.0));
        assert!(matches!(&splits[1][0], Value::Float(f) if *f == 42.0));
    }

    #[test]
    fn split_tensor_dim0_mismatch_broadcasts() {
        let tensor = Value::Tensor(Tensor {
            data: TensorData::Float32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![4, 1],
        });
        let splits = split_results(vec![tensor], &[2, 1]);
        match (&splits[0][0], &splits[1][0]) {
            (Value::Tensor(a), Value::Tensor(b)) => {
                assert_eq!(a.shape, vec![4, 1]);
                assert_eq!(b.shape, vec![4, 1]);
            }
            _ => panic!("expected tensors"),
        }
    }

    #[test]
    fn merge_split_round_trip() {
        // Two requests of 2 + 1 items merge into a 3-item invocation whose
        // list output splits back to the original request shapes.
        let bp: HashSet<String> = HashSet::from(["texts".into()]);
        let a = HashMap::from([
            ("texts".to_string(), Value::List(vec!["a".into(), "b".into()])),
        ]);
        let b = HashMap::from([
            ("texts".to_string(), Value::List(vec!["c".into()])),
        ]);
        let merged = merge_inputs(&[&a, &b], &bp);
        let merged_texts = match merged.get("texts") {
            Some(Value::List(items)) => items.clone(),
            other => panic!("expected list, got {other:?}"),
        };
        let splits = split_results(vec![Value::List(merged_texts)], &[2, 1]);
        assert!(matches!(&splits[0][0], Value::List(items) if items.len() == 2));
        assert!(matches!(&splits[1][0], Value::List(items) if items.len() == 1));
    }
}
