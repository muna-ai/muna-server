/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Validate bounded HTTP ranges before publishing any resource.

use std::fs::File;
use std::sync::Arc;

use muna::DownloadProgressFn;
use reqwest::header::{
    ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_RANGE, ETAG, IF_RANGE, LAST_MODIFIED, RANGE,
};
use reqwest::{Client, Response, StatusCode};

use super::file;

#[derive(Clone)]
pub(super) struct Validator(pub reqwest::header::HeaderValue);

pub(super) fn validator(response: &Response) -> Option<Validator> {
    response
        .headers()
        .get(ETAG)
        .filter(|value| !value.as_bytes().starts_with(b"W/"))
        .or_else(|| response.headers().get(LAST_MODIFIED))
        .cloned()
        .map(Validator)
}

pub(super) fn content_range(response: &Response) -> Option<(u64, u64, u64)> {
    let value = response.headers().get(CONTENT_RANGE)?.to_str().ok()?;
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()?, total.parse().ok()?))
}

pub(super) fn identity(response: &Response) -> Result<(), String> {
    if response
        .headers()
        .get(CONTENT_ENCODING)
        .is_some_and(|value| value != "identity")
    {
        return Err("Resource response must use identity content encoding.".into());
    }
    Ok(())
}

pub(super) async fn download(
    http: &Client,
    url: &str,
    file: Arc<File>,
    range: std::ops::Range<u64>,
    total: u64,
    validator: Option<Validator>,
    progress: DownloadProgressFn,
) -> Result<(), String> {
    let mut request = http
        .get(url)
        .header(ACCEPT_ENCODING, "identity")
        .header(RANGE, format!("bytes={}-{}", range.start, range.end - 1));
    if let Some(validator) = validator {
        request = request.header(IF_RANGE, validator.0);
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| error.without_url().to_string())?;
    identity(&response)?;
    if response.status() != StatusCode::PARTIAL_CONTENT
        || content_range(&response) != Some((range.start, range.end - 1, total))
    {
        return Err(
            "Resource returned an unexpected byte range or changed during download.".into(),
        );
    }
    let expected = range.end - range.start;
    if response
        .content_length()
        .is_some_and(|length| length != expected)
    {
        return Err("Resource range has an unexpected length.".into());
    }
    // One bounded buffer and one blocking write per range, rather than a
    // blocking task per network packet. All buffers fit in the current quantum.
    let mut bytes = Vec::with_capacity(expected as usize);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| error.without_url().to_string())?
    {
        if chunk.len() as u64 > expected.saturating_sub(bytes.len() as u64) {
            return Err("Resource range exceeded its declared length.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    if bytes.len() as u64 != expected {
        return Err("Resource range ended before its declared length.".into());
    }
    let written = file::write(file, bytes, range.start).await?;
    progress(written, Some(total));
    Ok(())
}
