/*
*   Muna
*   Copyright © 2026 NatML Inc. All Rights Reserved.
*/

//! Remote image inlining for the chat surfaces.
//!
//! muna-rs decodes images carried inline in the request and never fetches
//! them, so the server fetches remote images and rewrites them in place
//! before calling muna-rs: OpenAI `image_url` parts become `data:` URLs and
//! Anthropic `url` image sources become `base64` sources.
//!
//! Requests come from untrusted clients, so the fetcher only reaches public
//! addresses: hostnames resolve through a resolver that drops non-global
//! addresses, and IP-literal hosts (which skip the resolver) are checked on
//! the initial URL and on every redirect.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use futures_util::{stream, StreamExt};
use muna::beta::anthropic::{ContentBlockParam, ImageSource, MessageContent, MessageCreateParams};
use muna::beta::openai::{ChatCompletionContent, ChatCompletionContentPart, ChatCompletionCreateParams};
use muna::MunaError;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::redirect::Policy;
use reqwest::Url;

/// Maximum size of one fetched image.
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Maximum total bytes fetched for one request. Matches the inference
/// routes' request body limit: no more than the client could have sent
/// inline.
const MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of redirects followed per image.
const MAX_REDIRECTS: usize = 3;

/// Maximum number of images fetched concurrently for one request.
const MAX_CONCURRENT_FETCHES: usize = 4;

/// Fetches remote images for inlining. Fetch failures are caller errors
/// (`MunaError::InvalidInput`, a 400) naming the image URL.
pub(crate) struct ImageFetcher {
    http: reqwest::Client,
    max_image_bytes: usize,
    allow_private: bool,
}

/// A fetched image.
struct FetchedImage {
    data: Vec<u8>,
    mime: String,
}

impl ImageFetcher {

    /// Create a fetcher that only reaches public addresses.
    pub(crate) fn new() -> Self {
        Self::build(MAX_IMAGE_BYTES, false)
    }

    fn build(
        max_image_bytes: usize,
        allow_private: bool
    ) -> Self {
        let redirect = Policy::custom(move |attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                return attempt.error(format!("more than {MAX_REDIRECTS} redirects"));
            }
            match check_url(attempt.url(), allow_private) {
                Ok(()) => attempt.follow(),
                Err(reason) => attempt.error(reason),
            }
        });
        // System proxies resolve hostnames themselves, which would bypass
        // the public-address resolver.
        let mut builder = reqwest::Client::builder()
            .user_agent("muna-server")
            .no_proxy()
            .redirect(redirect)
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30));
        if !allow_private {
            builder = builder.dns_resolver(Arc::new(PublicResolver));
        }
        Self {
            http: builder.build().expect("failed to build reqwest client"),
            max_image_bytes,
            allow_private,
        }
    }

    /// Fetch remote images in OpenAI `image_url` parts and rewrite them as
    /// `data:` URLs. `data:` URLs are left untouched.
    pub(crate) async fn inline_openai(
        &self,
        params: &mut ChatCompletionCreateParams
    ) -> Result<(), MunaError> {
        let mut targets: Vec<&mut String> = params
            .messages
            .iter_mut()
            .filter_map(|message| match &mut message.content {
                Some(ChatCompletionContent::Parts(parts)) => Some(parts.iter_mut()),
                _ => None,
            })
            .flatten()
            .filter_map(|part| match part {
                ChatCompletionContentPart::ImageUrl { image_url }
                    if !image_url.url.starts_with("data:") => Some(&mut image_url.url),
                _ => None,
            })
            .collect();
        if targets.is_empty() {
            return Ok(());
        }
        let urls = targets.iter().map(|url| url.to_string()).collect();
        let images = self.fetch_all(urls).await?;
        for (target, image) in targets.iter_mut().zip(images) {
            **target = format!("data:{};base64,{}", image.mime, BASE64.encode(&image.data));
        }
        Ok(())
    }

    /// Fetch remote images in Anthropic `url` image sources and rewrite
    /// them as `base64` sources. Only top-level message blocks are
    /// rewritten; muna-rs rejects images anywhere else.
    pub(crate) async fn inline_anthropic(
        &self,
        params: &mut MessageCreateParams
    ) -> Result<(), MunaError> {
        let mut targets: Vec<&mut ImageSource> = params
            .messages
            .iter_mut()
            .filter_map(|message| match &mut message.content {
                MessageContent::Blocks(blocks) => Some(blocks.iter_mut()),
                MessageContent::Text(_) => None,
            })
            .flatten()
            .filter_map(|block| match block {
                ContentBlockParam::Image { source }
                    if matches!(source, ImageSource::Url { .. }) => Some(source),
                _ => None,
            })
            .collect();
        if targets.is_empty() {
            return Ok(());
        }
        let urls = targets
            .iter()
            .filter_map(|source| match source {
                ImageSource::Url { url } => Some(url.clone()),
                _ => None,
            })
            .collect();
        let images = self.fetch_all(urls).await?;
        for (target, image) in targets.iter_mut().zip(images) {
            **target = ImageSource::Base64 {
                media_type: image.mime,
                data: BASE64.encode(&image.data),
            };
        }
        Ok(())
    }

    /// Fetch images concurrently, in order, within the per-request budget.
    async fn fetch_all(
        &self,
        urls: Vec<String>
    ) -> Result<Vec<FetchedImage>, MunaError> {
        let mut fetches = stream::iter(urls)
            .map(|url| async move {
                self.fetch(&url).await.map_err(|reason| MunaError::InvalidInput(
                    format!("Failed to fetch image at {url}: {reason}")
                ))
            })
            .buffered(MAX_CONCURRENT_FETCHES);
        let mut images = Vec::new();
        let mut total = 0;
        while let Some(image) = fetches.next().await {
            let image = image?;
            total += image.data.len();
            if total > MAX_TOTAL_BYTES {
                return Err(MunaError::InvalidInput(format!(
                    "Remote images exceed the {} MB total limit per request.",
                    MAX_TOTAL_BYTES / (1024 * 1024)
                )));
            }
            images.push(image);
        }
        Ok(images)
    }

    async fn fetch(&self, url: &str) -> Result<FetchedImage, String> {
        let url = Url::parse(url).map_err(|e| format!("invalid URL ({e})"))?;
        check_url(&url, self.allow_private)?;
        let mut response = self.http.get(url).send().await.map_err(|e| error_chain(&e))?;
        if !response.status().is_success() {
            return Err(format!("server responded with {}", response.status()));
        }
        let mime = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .map(|value| value.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if !mime.starts_with("image/") {
            return Err(format!("content type `{mime}` is not an image"));
        }
        let too_large = || format!("image exceeds the {} MB limit", self.max_image_bytes / (1024 * 1024));
        if response.content_length().is_some_and(|length| length > self.max_image_bytes as u64) {
            return Err(too_large());
        }
        let mut data = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|e| error_chain(&e))? {
            if data.len() + chunk.len() > self.max_image_bytes {
                return Err(too_large());
            }
            data.extend_from_slice(&chunk);
        }
        Ok(FetchedImage { data, mime })
    }
}

/// Resolver that only returns public addresses.
struct PublicResolver;

impl Resolve for PublicResolver {

    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((name.as_str(), 0))
                .await?
                .filter(|addr| is_public_ip(addr.ip()))
                .collect();
            if addrs.is_empty() {
                return Err(format!("{} does not resolve to a public address", name.as_str()).into());
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

/// Accept `http` / `https` URLs whose host is a hostname (checked at
/// resolution) or a public IP literal.
fn check_url(
    url: &Url,
    allow_private: bool
) -> Result<(), String> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("unsupported URL scheme `{}`", url.scheme()));
    }
    let host = url.host_str().ok_or("URL has no host")?;
    let literal = host.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>();
    match literal {
        Ok(ip) if !allow_private && !is_public_ip(ip) => Err(format!("{ip} is not a public address")),
        _ => Ok(()),
    }
}

/// Whether an address is globally routable (`std`'s `is_global` is
/// unstable).
fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_ipv4(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public_ipv4(v4);
            }
            let s = v6.segments();
            !(
                s[..6] == [0; 6]                        ||  // unspecified, loopback, IPv4-compatible
                v6.is_multicast()                       ||
                (s[0] & 0xfe00) == 0xfc00               ||  // unique local fc00::/7
                (s[0] & 0xffc0) == 0xfe80               ||  // link-local fe80::/10
                (s[0] == 0x2001 && s[1] == 0x0db8)      ||  // documentation 2001:db8::/32
                (s[0] == 0x0064 && s[1] == 0xff9b)          // NAT64 64:ff9b::/96 can embed private IPv4
            )
        }
    }
}

fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(
        a == 0                                  ||  // this network
        a == 10                                 ||  // private
        a == 127                                ||  // loopback
        a >= 224                                ||  // multicast, reserved, broadcast
        (a == 100 && (b & 0xc0) == 64)          ||  // CGNAT 100.64/10
        (a == 169 && b == 254)                  ||  // link-local, cloud metadata
        (a == 172 && (b & 0xf0) == 16)          ||  // private 172.16/12
        (a == 192 && b == 168)                  ||  // private
        (a == 192 && b == 0 && (c == 0 || c == 2)) ||  // protocol assignments, TEST-NET-1
        (a == 198 && (b & 0xfe) == 18)          ||  // benchmarking 198.18/15
        (a == 198 && b == 51 && c == 100)       ||  // TEST-NET-2
        (a == 203 && b == 0 && c == 113)            // TEST-NET-3
    )
}

/// Render an error with its sources: reqwest's own message omits the cause
/// (e.g. why a redirect or resolution was refused).
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    message
}

#[cfg(test)]
mod tests {
    use axum::http::{header, StatusCode};
    use axum::response::{IntoResponse, Redirect};
    use axum::routing::get;
    use axum::Router;
    use serde_json::json;

    use super::*;

    /// 2x2 solid-red RGBA PNG.
    const RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8z8Dwn4GBgYEJRIAwAB8XAgICR7MUAAAAAElFTkSuQmCC";

    /// Serve test images on loopback; returns the base URL.
    async fn serve() -> String {
        let png = BASE64.decode(RED_PNG_B64).unwrap();
        let big = vec![0u8; 4096];
        let app = Router::new()
            .route("/red.png", get(move || async move {
                ([(header::CONTENT_TYPE, "image/png")], png).into_response()
            }))
            .route("/big.png", get(move || async move {
                ([(header::CONTENT_TYPE, "image/png")], big).into_response()
            }))
            .route("/page.html", get(|| async {
                ([(header::CONTENT_TYPE, "text/html")], "<html></html>").into_response()
            }))
            .route("/missing.png", get(|| async { StatusCode::NOT_FOUND }))
            .route("/redirect", get(|| async { Redirect::temporary("/red.png") }))
            .route("/loop", get(|| async { Redirect::temporary("/loop") }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// Loopback-permissive fetcher with a 1 KB image cap.
    fn test_fetcher() -> ImageFetcher {
        ImageFetcher::build(1024, true)
    }

    fn openai_params(urls: &[String]) -> ChatCompletionCreateParams {
        let mut parts = vec![json!({ "type": "text", "text": "describe" })];
        parts.extend(urls.iter().map(|url| json!({ "type": "image_url", "image_url": { "url": url } })));
        serde_json::from_value(json!({
            "model": "@a/x",
            "messages": [{ "role": "user", "content": parts }],
        })).unwrap()
    }

    fn image_urls(params: &ChatCompletionCreateParams) -> Vec<String> {
        let Some(ChatCompletionContent::Parts(parts)) = &params.messages[0].content else {
            panic!("expected parts content");
        };
        parts
            .iter()
            .filter_map(|part| match part {
                ChatCompletionContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn classifies_public_and_private_addresses() {
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "::ffff:8.8.8.8"] {
            assert!(is_public_ip(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "0.0.0.0", "10.0.0.1", "127.0.0.1", "100.64.0.1", "169.254.169.254",
            "172.16.0.1", "172.31.255.255", "192.168.1.1", "192.0.2.1", "198.18.0.1",
            "224.0.0.1", "255.255.255.255", "::", "::1", "::ffff:127.0.0.1",
            "::ffff:169.254.169.254", "fc00::1", "fd12::1", "fe80::1", "2001:db8::1",
            "64:ff9b::a00:1", "ff02::1"
        ] {
            assert!(!is_public_ip(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn check_url_rejects_schemes_and_private_literals() {
        let check = |url: &str| check_url(&Url::parse(url).unwrap(), false);
        assert!(check("https://example.com/cat.png").is_ok());
        assert!(check("http://8.8.8.8/cat.png").is_ok());
        assert!(check("file:///etc/passwd").is_err());
        assert!(check("ftp://example.com/cat.png").is_err());
        assert!(check("http://127.0.0.1/cat.png").is_err());
        assert!(check("http://169.254.169.254/latest/meta-data").is_err());
        assert!(check("http://[::1]:8080/cat.png").is_err());
        assert!(check("http://[::ffff:10.0.0.1]/cat.png").is_err());
    }

    #[tokio::test]
    async fn public_fetcher_refuses_loopback() {
        let base = serve().await;
        let fetcher = ImageFetcher::new();
        let error = fetcher.fetch(&format!("{base}/red.png")).await.err().unwrap();
        assert!(error.contains("not a public address"), "{error}");
        // Hostnames go through the resolver, which drops loopback.
        let port = base.rsplit(':').next().unwrap();
        let error = fetcher.fetch(&format!("http://localhost:{port}/red.png")).await.err().unwrap();
        assert!(error.contains("public address"), "{error}");
    }

    #[tokio::test]
    async fn fetch_enforces_status_type_size_and_redirect_limits() {
        let base = serve().await;
        let fetcher = test_fetcher();
        let image = fetcher.fetch(&format!("{base}/redirect")).await.unwrap();
        assert_eq!(image.mime, "image/png");
        assert_eq!(image.data, BASE64.decode(RED_PNG_B64).unwrap());
        for (path, expected) in [
            ("/big.png", "exceeds"),
            ("/page.html", "not an image"),
            ("/missing.png", "404"),
            ("/loop", "redirects"),
        ] {
            let error = fetcher.fetch(&format!("{base}{path}")).await.err().unwrap();
            assert!(error.contains(expected), "{path}: {error}");
        }
    }

    #[tokio::test]
    async fn openai_remote_parts_become_data_urls() {
        let base = serve().await;
        let inline = format!("data:image/png;base64,{RED_PNG_B64}");
        let mut params = openai_params(&[format!("{base}/red.png"), inline.clone()]);
        test_fetcher().inline_openai(&mut params).await.unwrap();
        assert_eq!(image_urls(&params), vec![inline.clone(), inline]);
        let images = params.chat_inputs().unwrap().images.unwrap();
        assert_eq!(images.len(), 2);
        // A failed fetch is a 400 naming the URL.
        let mut params = openai_params(&[format!("{base}/missing.png")]);
        assert!(test_fetcher().inline_openai(&mut params).await.is_err());
    }

    #[tokio::test]
    async fn anthropic_url_sources_become_base64_sources() {
        let base = serve().await;
        let mut params: MessageCreateParams = serde_json::from_value(json!({
            "model": "@a/x",
            "max_tokens": 64,
            "messages": [{ "role": "user", "content": [
                { "type": "image", "source": { "type": "url", "url": format!("{base}/red.png") } },
                { "type": "text", "text": "what is this?" }
            ] }],
        })).unwrap();
        test_fetcher().inline_anthropic(&mut params).await.unwrap();
        let MessageContent::Blocks(blocks) = &params.messages[0].content else {
            panic!("expected blocks content");
        };
        assert!(matches!(
            &blocks[0],
            ContentBlockParam::Image { source: ImageSource::Base64 { media_type, data } }
                if media_type == "image/png" && data == RED_PNG_B64
        ));
        assert_eq!(params.chat_inputs().unwrap().images.unwrap().len(), 1);
    }
}
