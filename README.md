# Muna Server

An OpenAI-compatible HTTP server backed by a local [Muna](https://muna.ai) predictor.

`POST /v1/chat/completions` runs the requested predictor through the `muna` crate with
`local_gpu` acceleration, relaying output as OpenAI-style JSON or SSE chunks. The model tag
travels in each request's `model` field, so one server can serve any predictor on demand.

## Endpoints

| Method | Path                   | Description                                              |
| ------ | ---------------------- | -------------------------------------------------------- |
| `GET`  | `/` , `/health`        | Health check (`{"status":"ok"}`).                        |
| `GET`  | `/v1/models`           | Lists models this process has loaded so far.             |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion (JSON or SSE stream).  |

## Setup

The `muna` crate links a native `libFunction.so` (fetched by its `build.rs` from `cdn.fxn.ai`),
so the binary needs that library reachable at runtime via `LD_LIBRARY_PATH`. Muna reads the
access key from `$MUNA_ACCESS_KEY`.

```bash
# Build (downloads libFunction.so into target/.../out/ as a side effect)
cargo build

# Point the loader at the bundled libFunction.so
export LD_LIBRARY_PATH="$(dirname "$(find target -name libFunction.so | head -1)")"

# Provide your Muna access key (or put it in a .env file)
export MUNA_ACCESS_KEY=fxn_...
```

## Run

```bash
# Start the server (defaults to PORT=8000)
cargo run -- serve

# Optionally warm a predictor ahead of time so the first request is fast
cargo run -- preload @huggingface/smollm2-360m
```

## Request example

The `model` field is the Muna predictor tag — e.g. `@huggingface/smollm2-360m`.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "@huggingface/smollm2-360m",
    "messages": [{"role": "user", "content": "In one sentence, what is a llama?"}],
    "max_tokens": 64
  }'
```

Response:

```json
{
  "object": "chat.completion",
  "id": "fxn-705482",
  "model": "Smollm2 360M 8k Lc100K Mix1 Ep2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A llama is a large, hoofed domesticated mammal native to South America, known for its long, soft hair, thick coat, and ability to carry loads up to 330 pounds."
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "created": 1782319272,
  "usage": { "prompt_tokens": 41, "completion_tokens": 41, "total_tokens": 82 }
}
```

The response's `model` field is the predictor's internal display name, not the tag you passed.
The first request to a tag downloads and loads the model, so it is slower than warm requests;
use `preload` to avoid that latency.

### Streaming

Set `"stream": true` to receive Server-Sent Events (`data: {...chat.completion.chunk...}`),
terminated by a `data: [DONE]` event:

```bash
curl -s -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "@huggingface/smollm2-360m",
    "messages": [{"role": "user", "content": "Count to three."}],
    "stream": true,
    "max_tokens": 32
  }'
```
