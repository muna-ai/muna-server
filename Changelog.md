## 0.0.8
+ Minor updates.

## 0.0.7
+ Added support for Anthropic-compatible messages at `/v1/messages`.
+ Upgraded `muna` to 0.0.17.

## 0.0.6
+ Removed the `preload` subcommand. Pin models with `--models` to load them eagerly at boot instead.
+ Removed the `serve` subcommand. Serving is now the sole top-level command: `muna-server [flags]`.

## 0.0.5
+ Added `reasoning_content` passthrough for reasoning models in OpenAI-compatible chat completions.
+ Added download progress reporting for model resources.
+ Fixed parallel-safe resource downloads: concurrent model loads sharing a resource file no longer race.
+ Updated requests for a model that is still loading to return `429` with `Retry-After` instead of `503`.
+ Updated requests for models not in the `--models` allowlist to return `404` with OpenAI's `model_not_found` error shape.
+ Updated model loading to begin eagerly at boot for models pinned via `--models`.
+ Renamed server environment variables to the `MUNA_SERVER_` prefix (e.g. `MUNA_SERVER_MODELS`, `MUNA_SERVER_ID`, `MUNA_SERVER_TOKEN`).
+ Upgraded `muna` to 0.0.16.

## 0.0.4
+ Added `--models` CLI option to eagerly load and restrict the models that can be served.

## 0.0.3
+ Added support for OpenAI-compatible image generation at `/v1/images/generations`.
+ Added support for KV-aware routed compiled models.
+ Added support for invoking compiled models that use continuous, static, and dynamic batching.
+ Fixed model inference being synchronized across all requests (#2).

## 0.0.2
+ Upgraded `muna` to 0.0.12.

## 0.0.1
+ First pre-release.