/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Resumable transfers. A scheduling turn writes a bounded quantum into the
//! same temporary file; only a complete resource is renamed into the cache.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use futures_util::future::join_all;
use muna::DownloadProgressFn;
use reqwest::header::{ACCEPT_ENCODING, CONTENT_RANGE, RANGE};
use reqwest::{Client, Response, StatusCode};

mod file;
mod ranges;

#[derive(Clone, Copy)]
pub(super) struct Limits {
    pub range: u64,
    pub quantum: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            range: 4 * 1024 * 1024,
            quantum: 32 * 1024 * 1024,
        }
    }
}

enum Source {
    Ranges {
        validator: Option<ranges::Validator>,
    },
    Stream {
        response: Response,
        pending: Bytes,
    },
}

pub(super) struct Transfer {
    http: Client,
    url: String,
    destination: PathBuf,
    temporary: PathBuf,
    file: Option<Arc<std::fs::File>>,
    source: Option<Source>,
    total: Option<u64>,
    expected: Option<u64>,
    completed: u64,
}

impl Transfer {
    pub fn new(http: Client, url: String, destination: &Path) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = destination
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        let temporary = destination.with_file_name(format!(
            ".{name}.{}.{}.part",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        Self {
            http,
            url,
            destination: destination.into(),
            temporary,
            file: None,
            source: None,
            total: None,
            expected: None,
            completed: 0,
        }
    }

    pub fn with_size(mut self, expected: Option<u64>) -> Self {
        self.expected = expected;
        self
    }

    async fn open(&mut self, progress: &DownloadProgressFn) -> Result<(), String> {
        let mut response = self
            .http
            .get(&self.url)
            .header(ACCEPT_ENCODING, "identity")
            .header(RANGE, "bytes=0-0")
            .send()
            .await
            .map_err(|error| error.without_url().to_string())?;
        ranges::identity(&response)?;
        self.source = Some(if response.status() == StatusCode::PARTIAL_CONTENT {
            let Some((0, 0, total)) =
                ranges::content_range(&response).filter(|(_, _, total)| *total > 0)
            else {
                return Err("Resource probe returned an invalid byte range.".into());
            };
            self.total = Some(total);
            let validator = ranges::validator(&response);
            // Bound even the probe response: never collect an untrusted body.
            let mut received = 0;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|error| error.without_url().to_string())?
            {
                received += chunk.len();
                if received > 1 {
                    return Err("Resource probe exceeded one byte.".into());
                }
            }
            if received != 1 {
                return Err("Resource probe ended before one byte.".into());
            }
            Source::Ranges { validator }
        } else if response.status() == StatusCode::RANGE_NOT_SATISFIABLE
            && response
                .headers()
                .get(CONTENT_RANGE)
                .is_some_and(|value| value == "bytes */0")
        {
            self.total = Some(0);
            Source::Ranges { validator: None }
        } else if response.status() == StatusCode::OK {
            self.total = response.content_length();
            // Reuse this response when the origin ignores Range. Pausing means
            // retaining it and applying HTTP backpressure until the next turn.
            Source::Stream {
                response,
                pending: Bytes::new(),
            }
        } else {
            return Err(format!(
                "Resource download returned HTTP {}.",
                response.status()
            ));
        });
        if self
            .total
            .zip(self.expected)
            .is_some_and(|(actual, expected)| actual != expected)
        {
            return Err("Resource size differs from its admitted manifest.".into());
        }
        if let Some(parent) = self
            .destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| error.to_string())?;
        }
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.temporary)
            .await
            .map_err(|error| error.to_string())?;
        if let Some(total) = self.total {
            file.set_len(total)
                .await
                .map_err(|error| error.to_string())?;
        }
        self.file = Some(Arc::new(file.into_std().await));
        progress(0, self.total);
        Ok(())
    }

    pub async fn advance(
        &mut self,
        limits: Limits,
        progress: DownloadProgressFn,
    ) -> Result<bool, String> {
        if self.file.is_none() {
            self.open(&progress).await?;
        }
        if self.total == Some(self.completed) {
            return Ok(true);
        }
        let file = self.file.as_ref().unwrap().clone();
        match self.source.as_mut().unwrap() {
            Source::Ranges { validator } => {
                let total = self.total.unwrap();
                let end = self.completed.saturating_add(limits.quantum).min(total);
                let mut next = self.completed;
                let mut requests = Vec::new();
                while next < end {
                    let stop = next.saturating_add(limits.range).min(end);
                    requests.push(ranges::download(
                        &self.http,
                        &self.url,
                        file.clone(),
                        next..stop,
                        total,
                        validator.clone(),
                        progress.clone(),
                    ));
                    next = stop;
                }
                // Drain every request, including on error, before cleaning up
                // the file or giving the next job the physical transfer slot.
                for result in join_all(requests).await {
                    result?;
                }
                self.completed = end;
                Ok(end == total)
            }
            Source::Stream { response, pending } => {
                let mut bytes = Vec::new();
                let mut eof = false;
                while bytes.len() < limits.quantum as usize {
                    if pending.is_empty() {
                        match response
                            .chunk()
                            .await
                            .map_err(|error| error.without_url().to_string())?
                        {
                            Some(chunk) => *pending = chunk,
                            None => {
                                eof = true;
                                break;
                            }
                        }
                    }
                    let count = pending.len().min(limits.quantum as usize - bytes.len());
                    bytes.extend_from_slice(&pending.split_to(count));
                }
                let next = self.completed.saturating_add(bytes.len() as u64);
                if self
                    .total
                    .or(self.expected)
                    .is_some_and(|total| next > total || (eof && next != total))
                {
                    return Err("Resource stream has an unexpected length.".into());
                }
                let written = file::write(file, bytes, self.completed).await?;
                self.completed = next;
                progress(written, self.total);
                Ok(eof || self.total == Some(next))
            }
        }
    }

    pub async fn publish(&mut self) -> Result<(), String> {
        self.source = None;
        self.file = None;
        // Ephemeral node cache: bytes are readable immediately from page cache.
        // Deliberately do not fsync every resource before making it available.
        tokio::fs::rename(&self.temporary, &self.destination)
            .await
            .map_err(|error| error.to_string())
    }

    pub async fn discard(&mut self) {
        self.source = None;
        self.file = None;
        let _ = tokio::fs::remove_file(&self.temporary).await;
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        // Normal success/error paths close the file first; this also covers an
        // unexpected task panic. Published resources have no temporary path.
        self.source = None;
        self.file = None;
        let _ = std::fs::remove_file(&self.temporary);
    }
}
