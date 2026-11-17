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
use tokio_util::sync::CancellationToken;

use crate::client::ServerClient;
use super::predict;
use super::downloads::{Downloads, Priority};
use super::preparation::PreparationTracker;

const MAX_AGE: Duration = Duration::from_secs(300);

/// Priority and cancellation follow the preparation, even while its caller
/// waits on a blocking worker. Retirement never cancels a native invocation.
#[derive(Default)]
struct Preparation {
    snapshot: Mutex<Option<Snapshot>>,
    priority: Priority,
    cancellation: CancellationToken,
}

impl Preparation {
    async fn run<T>(
        &self,
        operation: impl std::future::Future<Output = Result<T, muna::MunaError>>
    ) -> Result<T, muna::MunaError> {
        tokio::select! {
            biased;
            _ = self.cancellation.cancelled() => Err(muna::MunaError::Native("Model preparation retired.".into())),
            result = operation => result,
        }
    }
}

type Slot = Arc<Preparation>;

#[derive(Clone, Default)]
pub(crate) struct PreparedModels {
    slots: Arc<DashMap<String, Slot>>,
    downloads: Downloads,
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

    pub(crate) fn set_process(&self, tag: &str, process: bool) {
        self.slots.entry(tag.to_owned()).or_default().priority.set_process(process);
    }

    pub(crate) async fn prepare(
        &self,
        tag: &str,
        key: Option<String>,
        progress: PreparationTracker
    ) -> Result<Arc<PreparedModel>, String> {
        let slot = self.slots.entry(tag.to_owned()).or_default().clone();
        let mut snapshot = slot.snapshot.lock().await;
        if let Some(cached) = snapshot.as_ref().filter(|cached| cached.reusable(&key)) {
            return Ok(cached.model.clone());
        }
        let priority = slot.priority.clone();
        let transport = Arc::new(ServerClient::scheduled(
            key.clone(),
            tag,
            progress,
            self.downloads.clone(),
            priority
        ));
        let client = Arc::new(Muna::with_client(transport.clone()));
        let tag = tag.to_owned();
        let preparation = slot.clone();
        let (prediction, signature) = predict::run(None, move || async move {
            preparation.run(async move {
                let prediction = client.predictions.create(
                    &tag,
                    Some(HashMap::<String, Value>::new()),
                    Some(Acceleration::LocalAuto),
                    None,
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
            }).await
        }).await.map_err(|error| error.to_string())?;
        let model = Arc::new(PreparedModel { client: transport, prediction, signature });
        *snapshot = Some(Snapshot { key, at: Instant::now(), model: model.clone() });
        Ok(model)
    }

    pub(crate) fn forget(&self, tag: &str) {
        // An in-flight preparation may finish in its detached slot, but cannot
        // reinsert itself after retirement or supply a replacement deployment.
        if let Some((_, preparation)) = self.slots.remove(tag) {
            preparation.cancellation.cancel();
        }
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
        models.slots.insert("@a/cpu".into(), Arc::new(Preparation { snapshot: Mutex::new(Some(cached)), ..Preparation::default() }));
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
        let old = Arc::new(Preparation::default());
        models.slots.insert("@a/cpu".into(), old.clone());
        models.forget("@a/cpu");
        assert!(old.cancellation.is_cancelled());
        let replacement = models.slots.entry("@a/cpu".into()).or_default().clone();
        assert!(!replacement.cancellation.is_cancelled());
        *old.snapshot.lock().await = Some(snapshot(None));
        assert!(replacement.snapshot.lock().await.is_none());
    }
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[tokio::test]
    async fn retirement_stops_a_preparation_inside_the_blocking_worker() {
        let models = PreparedModels::default();
        let slot = models.slots.entry("retired".into()).or_default().clone();
        let started = Arc::new(tokio::sync::Notify::new());
        let worker_started = started.clone();
        let task = tokio::spawn(predict::run(None, move || async move {
            slot.run(async move {
                worker_started.notify_one();
                std::future::pending::<Result<(), muna::MunaError>>().await
            }).await
        }));
        started.notified().await;
        models.forget("retired");
        let result = tokio::time::timeout(Duration::from_secs(3), task).await.unwrap().unwrap();
        assert!(result.unwrap_err().to_string().contains("retired"));
    }
}
