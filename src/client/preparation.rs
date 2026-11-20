/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Admit the exact manifest returned to the SDK, before it starts any transfer.

use std::collections::HashMap;

use muna::types::PredictionResource;
use muna::MunaError;
use reqwest::header::CONTENT_LENGTH;

use super::ServerClient;
use crate::serving::preparation::{FailureStage, PreparationFailure, PreparationTracker};

impl ServerClient {
    pub(super) async fn admit(
        &self,
        tag: &str,
        tracker: &PreparationTracker,
        body: &serde_json::Value,
    ) -> Result<(), MunaError> {
        let gpu_count = body
            .pointer("/requirements/gpuCount")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let gpu_count =
            u32::try_from(gpu_count).map_err(|_| MunaError::Native("Invalid GPU count.".into()))?;
        if self
            .available_gpus
            .is_some_and(|available| gpu_count > available)
        {
            tracker.rejected(tag, PreparationFailure::incompatible());
            return Err(MunaError::Native(
                "This implementation requires more GPUs than this node has.".into(),
            ));
        }
        let mut resources: Vec<PredictionResource> = serde_json::from_value(
            body.get("resources")
                .cloned()
                .unwrap_or(serde_json::json!([])),
        )
        .map_err(|error| MunaError::Native(format!("Invalid resource manifest: {error}")))?;
        // The API normally supplies sizes. Legacy entries can use the local
        // cache or HEAD; unknown sizes must never silently count as zero.
        for resource in &mut resources {
            if resource.size.is_none() {
                let (_, path) =
                    crate::serving::preparation::resource_path(self.inner.cache_path(), resource);
                resource.size = match tokio::fs::metadata(path).await {
                    Ok(metadata) if metadata.is_file() => Some(metadata.len()),
                    _ => self
                        .http
                        .head(&resource.url)
                        .send()
                        .await
                        .ok()
                        .filter(|response| response.status().is_success())
                        .and_then(|response| {
                            // HEAD has no body: content_length() can report zero
                            // even when the header describes a nonempty resource.
                            response.headers().get(CONTENT_LENGTH)?.to_str().ok()?.parse().ok()
                        }),
                };
            }
        }
        tracker
            .manifest(tag, self.inner.cache_path(), &resources)
            .await;
        tracker.resolved(tag, gpu_count, &resources);
        let mut manifest = HashMap::new();
        for resource in &resources {
            let (_, path) =
                crate::serving::preparation::resource_path(self.inner.cache_path(), resource);
            let size = resource.size.ok_or_else(|| {
                MunaError::Native("Cannot determine resource size for disk admission.".into())
            })?;
            if manifest
                .insert(path, size)
                .is_some_and(|previous| previous != size)
            {
                return Err(MunaError::Native("Conflicting resource sizes.".into()));
            }
        }
        let reservation = self
            .downloads
            .disk()
            .reserve(self.inner.cache_path(), &manifest, |waiting| {
                tracker.waiting(tag, waiting)
            })
            .await
            .map_err(|error| {
                tracker.failed(tag, FailureStage::Download);
                MunaError::Native(error)
            })?;
        self.reservations.lock().unwrap().push(reservation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notifications::NotificationCenter;
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn head_resource(response: String) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let url = format!("http://{address}/head-resource-{}", address.port());
        let server = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                    assert_ne!(stream.read_buf(&mut request).await.unwrap(), 0);
                }
                assert!(request.starts_with(b"HEAD "));
                stream.write_all(response.as_bytes()).await.unwrap();
            }).await.unwrap();
        });
        (url, server)
    }

    #[tokio::test]
    async fn head_header_size_drives_progress_and_shared_disk_reservations() {
        for size in [0_u64, 8192] {
            // Send only headers, as a real HEAD response does. A zero-length
            // resource is valid only when the server explicitly declares it.
            let (url, server) = head_resource(format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n"
            )).await;
            let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
            let client = ServerClient::tracking(None, "model", tracker.clone());
            let mut body = serde_json::json!({"resources": [{"type": "bin", "url": url}]});
            client.admit("model", &tracker, &body).await.unwrap();
            server.await.unwrap();
            assert_eq!(tracker.snapshot("model").unwrap().bytes_total, Some(size));
            assert_eq!(tracker.requirements("model").unwrap().bin_bytes, Some(size));

            // Another deployment can know the size from the API. Both must
            // reserve the same shared file without conflicting size reports.
            body["resources"][0]["size"] = serde_json::json!(size);
            client.admit("other-model", &tracker, &body).await.unwrap();
            assert_eq!(client.reservations.lock().unwrap().len(), 2);
        }
    }

    #[tokio::test]
    async fn head_missing_invalid_and_unsuccessful_sizes_are_not_admitted_as_zero() {
        for response in [
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
            "HTTP/1.1 200 OK\r\nContent-Length: invalid\r\nConnection: close\r\n\r\n",
            "HTTP/1.1 404 Not Found\r\nContent-Length: 8192\r\nConnection: close\r\n\r\n",
        ] {
            let (url, server) = head_resource(response.into()).await;
            let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
            let client = ServerClient::tracking(None, "model", tracker.clone());
            let body = serde_json::json!({"resources": [{"type": "bin", "url": url}]});
            let error = client.admit("model", &tracker, &body).await.unwrap_err();
            server.await.unwrap();
            assert!(error.to_string().contains("Cannot determine resource size"));
            assert!(tracker.snapshot("model").unwrap().bytes_total.is_none());
            assert!(client.reservations.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn oversized_gpu_build_is_rejected_before_resolving_or_downloading_files() {
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let client =
            ServerClient::tracking(None, "model", tracker.clone()).with_gpu_capacity(Some(1));
        // Missing sizes would require an HTTP request if admission proceeded.
        let body = serde_json::json!({"requirements": {"gpuCount": 2}, "resources": [
            {"type": "bin", "url": "http://127.0.0.1:1/never-download"}
        ]});
        let error = client.admit("model", &tracker, &body).await.unwrap_err();
        assert!(error.to_string().contains("more GPUs"));
        let failure = tracker.failure("model").unwrap();
        assert!(!failure.retrying);
        assert_eq!(failure.code, "no_compatible_implementation");
        assert!(tracker.snapshot("model").unwrap().resources_total.is_none());
    }

    #[tokio::test]
    async fn manifest_reports_bin_sizes_and_retains_the_node_selected_build() {
        let tracker = PreparationTracker::new(Arc::new(NotificationCenter::default()));
        let client =
            ServerClient::tracking(None, "model", tracker.clone()).with_gpu_capacity(Some(2));
        let body = serde_json::json!({"requirements": {"gpuCount": 2}, "resources": [
            {"type": "bin", "url": "https://cdn.example/resources/admission-weights", "size": 32},
            {"type": "dso", "url": "https://cdn.example/resources/admission-code?signature=secret", "size": 1024}
        ]});
        client.admit("model", &tracker, &body).await.unwrap();
        let requirements = tracker.requirements("model").unwrap();
        assert_eq!(requirements.gpu_count, 2);
        assert_eq!(requirements.bin_bytes, Some(32));
        assert_eq!(tracker.build("model").as_deref(), Some("admission-code"));
        assert_eq!(tracker.snapshot("model").unwrap().bytes_total, Some(1056));
        assert_eq!(client.reservations.lock().unwrap().len(), 1);
    }
}
