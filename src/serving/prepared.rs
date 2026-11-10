/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Reuse disk preparation when promoting a predictor to process residency.
//! This owns metadata and a download-only client, never creates an engine.
//! Native loading still performs the SDK's normal token refresh and validation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use muna::types::{Acceleration, Prediction, Signature, Value};
use muna::Muna;
use tokio::sync::Mutex;

use crate::client::ServerClient;
use super::predict;
use super::preparation::PreparationTracker;

const MAX_AGE: Duration = Duration::from_secs(300);

type Slot = Arc<Mutex<Option<Snapshot>>>;

#[derive(Clone, Default)]
pub(crate) struct PreparedModels {
    slots: Arc<DashMap<String, Slot>>,
}

pub(crate) struct PreparedModel {
    // Keep only the transport here. Native handles belong exclusively to the
    // registry's Muna instance, so removing a registry slot still unloads it.
    pub client: Arc<ServerClient>,
    pub prediction: Prediction,
    pub signature: Signature,
}

struct Snapshot {
    key: Option<String>,
    at: Instant,
    model: Arc<PreparedModel>,
}

impl Snapshot {
    fn reusable(&self, key: &Option<String>) -> bool {
        self.key == *key && self.at.elapsed() < MAX_AGE
    }
}

impl PreparedModels {

    pub(crate) async fn prepare(
        &self,
        tag: &str,
        key: Option<String>,
        progress: PreparationTracker
    ) -> Result<Arc<PreparedModel>, String> {
        let slot = self.slots.entry(tag.to_owned()).or_default().clone();
        let mut snapshot = slot.lock().await;
        if let Some(cached) = snapshot.as_ref().filter(|cached| cached.reusable(&key)) {
            return Ok(cached.model.clone());
        }
        let transport = Arc::new(ServerClient::tracking(key.clone(), tag, progress));
        let client = Arc::new(Muna::with_client(transport.clone()));
        let tag = tag.to_owned();
        let (prediction, signature) = predict::run(None, move || async move {
            let prediction = client.predictions.create(
                &tag,
                Some(HashMap::<String, Value>::new()),
                Some(Acceleration::LocalAuto),
                None,
                None,
                None,
                None
            ).await?;
            if let Some(error) = &prediction.error {
                return Err(muna::MunaError::Prediction(error.clone()));
            }
            // Fetch the signature during disk preparation, outside the first
            // inference's engine-load path. No native predictor is constructed.
            let predictor = client.predictors.retrieve(&tag).await?
                .ok_or_else(|| muna::MunaError::Prediction(format!("predictor {tag} not found")))?;
            Ok((prediction, predictor.signature))
        }).await.map_err(|error| error.to_string())?;
        let model = Arc::new(PreparedModel { client: transport, prediction, signature });
        *snapshot = Some(Snapshot { key, at: Instant::now(), model: model.clone() });
        Ok(model)
    }

    pub(crate) fn forget(&self, tag: &str) {
        // An in-flight preparation may finish in its detached slot, but cannot
        // reinsert itself after retirement or supply a replacement deployment.
        self.slots.remove(tag);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notifications::NotificationCenter;

    fn snapshot(key: Option<String>) -> Snapshot {
        Snapshot {
            key,
            at: Instant::now(),
            model: Arc::new(PreparedModel {
                client: Arc::new(ServerClient::with_key(None)),
                prediction: serde_json::from_value(serde_json::json!({
                    "id":"prepared", "tag":"@a/cpu", "created":"2026-11-10T00:00:00Z"
                })).unwrap(),
                signature: Signature { inputs: vec![], outputs: vec![] },
            }),
        }
    }

    #[tokio::test]
    async fn disk_preparation_is_reused_without_an_api_request() {
        let models = PreparedModels::default();
        let cached = snapshot(Some("deployment-key".into()));
        let expected = cached.model.clone();
        models.slots.insert("@a/cpu".into(), Arc::new(Mutex::new(Some(cached))));
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let prepared = models.prepare("@a/cpu", Some("deployment-key".into()), tracker).await.unwrap();
        assert!(Arc::ptr_eq(&prepared, &expected));
        assert_eq!(prepared.prediction.id, "prepared");
    }

    #[test]
    fn rotated_credentials_and_expired_metadata_require_fresh_preparation() {
        let mut cached = snapshot(Some("old-deployment-key".into()));
        assert!(!cached.reusable(&Some("new-deployment-key".into())));
        assert!(!cached.reusable(&None));
        assert!(cached.reusable(&cached.key));
        cached.at -= MAX_AGE;
        assert!(!cached.reusable(&cached.key));
    }

    #[tokio::test]
    async fn late_preparation_cannot_reappear_after_retirement() {
        let models = PreparedModels::default();
        let old = Arc::new(Mutex::new(None));
        models.slots.insert("@a/cpu".into(), old.clone());
        models.forget("@a/cpu");
        let replacement = models.slots.entry("@a/cpu".into()).or_default().clone();
        *old.lock().await = Some(snapshot(None));
        assert!(replacement.lock().await.is_none());
    }
}
