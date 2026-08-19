# Fake Test Predictors

CPU-only fake predictors that impersonate model signatures so
`tests/serving.rs` can exercise muna-server's full serving path
(HTTP handler -> registry -> dispatcher -> muna FFI -> compiled binary)
without a GPU or real weights. Each script's `EXPLAINER` docstring
documents the contract it satisfies and the Rust tests that consume it.

| Script | Tag | Probes |
|---|---|---|
| `openai_chat.py` | `@muna/test-openai-chat` | Chat completions (create + SSE stream), `cached_tokens` usage plumbing |
| `openai_embeddings.py` | `@muna/test-openai-embeddings` | Embeddings shape, determinism, usage |
| `openai_image.py` | `@muna/test-openai-image` | Image generations (b64, one per prompt) |
| `batch_sequential.py` | `@muna/test-batch-sequential` | Sequential dispatch guard |
| `batch_static.py` | `@muna/test-batch-static` | Buffered merge/split (static) |
| `batch_dynamic.py` | `@muna/test-batch-dynamic` | Buffered merge/split (dynamic) |
| `batch_continuous.py` | `@muna/test-batch-continuous` | Continuous dispatch (no serialization) |
| `slow_coldstart.py` | `@muna/test-slow-coldstart` | Loading window: 429 + `Retry-After` |

## Compiling and pushing

One-time setup (the repo-root `requirements.txt` pins the only dependency):

```sh
pip install -r ../../requirements.txt
muna auth login <access key>   # needs push rights to the @muna organization
```

Compile + push all eight (each `@compile` decorator carries its tag):

```sh
for script in openai_chat openai_embeddings openai_image batch_sequential \
              batch_static batch_dynamic batch_continuous slow_coldstart; do
    muna compile --overwrite "$script.py"
done
```

Any script can be sanity-checked locally before compiling:

```sh
python openai_chat.py
```

The tags are hardcoded as constants in `tests/serving.rs`; recompiling
in place (`--overwrite`) is all that is needed to update a fake.
