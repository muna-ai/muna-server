/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Deployment-scoped API metadata. SDK caches may be read before request(), so
//! their directory must also be isolated. Resource files stay content-addressed
//! in the shared root and are linked into each SDK namespace on demand.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use muna::client::RequestInput;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

static WRITES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn key_hash(key: &str) -> String {
    format!("{:x}", Sha256::digest(key.as_bytes()))
}

pub(super) struct DeploymentCache {
    root: PathBuf,
    sdk: PathBuf,
    metadata: PathBuf,
}

impl DeploymentCache {
    pub fn new(root: &Path, key: &str) -> Self {
        let directory = root.join("deployments").join(key_hash(key));
        Self { root: root.into(), sdk: directory.join("sdk"), metadata: directory.join("metadata") }
    }

    pub fn sdk(&self) -> &Path { &self.sdk }

    pub fn shared_path(&self, path: &Path) -> PathBuf {
        path.strip_prefix(&self.sdk).map(|relative| self.root.join(relative)).unwrap_or_else(|_| path.into())
    }

    pub fn entry(&self, input: &RequestInput, api: &str) -> Option<PathBuf> {
        let base = input.base_url.as_deref().unwrap_or(api).trim_end_matches('/');
        if base != api.trim_end_matches('/') { return None; }
        let prediction = input.method == reqwest::Method::POST && input.path == "/predictions" &&
            input.body.as_ref().is_some_and(|body| body.get("tag").and_then(Value::as_str).is_some() &&
                body.get("inputs").is_none() && body.get("predictionId").is_none());
        let predictor = input.method == reqwest::Method::GET && input.path.starts_with("/predictors/");
        if !prediction && !predictor { return None; }
        // Normalize object ordering; every selector (target, architecture,
        // configuration identity, variants) remains part of the cache key.
        let identity = canonical(json!({ "api": base, "path": input.path, "body": input.body, "headers": input.headers }));
        let digest = Sha256::digest(serde_json::to_vec(&identity).ok()?);
        Some(self.metadata.join(format!("{digest:x}.json")))
    }

    pub async fn read(&self, path: &Path, input: &RequestInput) -> Option<Value> {
        let data = tokio::fs::read(path).await.ok()?;
        let body: Value = serde_json::from_slice(&data).ok()?;
        if input.method == reqwest::Method::GET {
            let _: muna::types::Predictor = serde_json::from_value(body.clone()).ok()?;
            return Some(body);
        }
        let prediction: muna::types::Prediction = serde_json::from_value(body.clone()).ok()?;
        if prediction.tag != input.body.as_ref()?.get("tag")?.as_str()? || prediction.error.is_some() {
            return None;
        }
        // Retain pinned metadata indefinitely, but never replay an expired
        // runtime token or expired download URLs for a missing file.
        if let Some(token) = &prediction.configuration {
            if let Some(expiry) = token.split('.').nth(1)
                .and_then(|payload| URL_SAFE_NO_PAD.decode(payload).ok())
                .and_then(|payload| serde_json::from_slice::<Value>(&payload).ok())
                .and_then(|claims| claims.get("exp").and_then(Value::as_u64)) {
                let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
                if now.saturating_add(60) >= expiry { return None; }
            }
        }
        for resource in prediction.resources.iter().flatten() {
            let url = reqwest::Url::parse(&resource.url).ok()?;
            let filename = url.path_segments()?.next_back()?;
            let mut resource_path = self.root.join(filename);
            if let Some(name) = &resource.name { resource_path.push(name); }
            let metadata = tokio::fs::metadata(resource_path).await.ok()?;
            if !metadata.is_file() || resource.size.is_some_and(|size| metadata.len() != size) { return None; }
        }
        Some(body)
    }

    pub async fn write(&self, path: &Path, body: &Value) {
        // Cache failure must not turn successful API resolution into a load
        // failure. Rename publishes complete JSON to other clients/processes.
        let Some(parent) = path.parent() else { return; };
        let Ok(data) = serde_json::to_vec(body) else { return; };
        if tokio::fs::create_dir_all(parent).await.is_err() { return; }
        let temporary = parent.join(format!(".{}.{}.tmp", std::process::id(), WRITES.fetch_add(1, Ordering::Relaxed)));
        if tokio::fs::write(&temporary, data).await.is_ok() {
            let _ = tokio::fs::rename(&temporary, path).await;
        }
        let _ = tokio::fs::remove_file(temporary).await;
    }
}

fn canonical(value: Value) -> Value {
    match value {
        Value::Object(fields) => {
            let fields: std::collections::BTreeMap<_, _> = fields.into_iter().collect();
            Value::Object(fields.into_iter().map(|(name, value)| (name, canonical(value))).collect())
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonical).collect()),
        value => value,
    }
}

#[cfg(test)]
#[path = "metadata_tests.rs"]
mod tests;
