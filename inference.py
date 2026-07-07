#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
This is the inference gateway that powers https://inference.muna.ai.

It is an OpenAI-compatible proxy that runs on Modal, forwarding to compiled
models we have deployed to our workspace with the `muna deploy` CLI.

Discovery is done at runtime:
1. Get all `muna-*` apps in our Modal workspace that have a running container.
2. `GET /v1/models` to see which models are alive on each one
3. Build a routing table from `tag -> URL`.
"""

from __future__ import annotations
from asyncio import create_task, gather, sleep, to_thread, wait_for
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from html import escape
from httpx import AsyncClient, HTTPError, Timeout
from modal import asgi_app, concurrent, App, Image
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2
from pydantic import BaseModel, Field, ValidationError
from string import Template
from time import monotonic, time
from typing import Generic, Literal, TypeVar

app = App("muna-inference-proxy")
image = Image.debian_slim().pip_install("fastapi[standard]", "httpx")

@app.function(
    image=image,
    min_containers=1,           # keep the gateway warm; clients never hit a cold proxy
    timeout=60 * 60,            # allow long streaming generations
    scaledown_window=300,       # wait this long before scaling down added replicas
    region="us-east",           # colocate with the backends to shorten the proxy->backend hop
)
@concurrent(max_inputs=100) # proxying is I/O-bound; one container fans out widely
@asgi_app(custom_domains=["inference.muna.ai"])
def muna_inference():
    # Create FastAPI app
    web_app = FastAPI(lifespan=_lifespan)
    # Landing page
    @web_app.get("/", response_class=HTMLResponse)
    async def landing(request: Request) -> HTMLResponse:
        return HTMLResponse(_render_landing(request.app.state.routes))
    # Health
    @web_app.get("/health")
    async def health(request: Request) -> dict:
        last_refresh = request.app.state.last_refresh
        return {
            "status": "ok",
            "model_count": len(request.app.state.routes),
            "last_refresh_age_s": round(monotonic() - last_refresh, 1) if last_refresh else None,
        }
    # GET /models
    @web_app.get("/v1/models")
    async def list_models(request: Request) -> Page[Model]:
        return Page[Model](
            data=[Model(id=tag) for tag in sorted(request.app.state.routes)],
        )
    # POST /chat/completions
    @web_app.post("/v1/chat/completions")
    async def create_chat_completion(request: Request):
        body = await request.json()
        model = body.get("model")
        if not model:
            return _error(
                "Missing required field `model`.",
                status=400,
                code="invalid_request"
            )
        urls = request.app.state.routes.get(model)
        if not urls:
            return _error(
                f"Model `{model}` is not available. Call GET /v1/models for the current list.",
                status=404,
                code="model_not_found",
            )
        return await _forward(
            f"{urls[0]}/v1/chat/completions",
            body,
            http=request.app.state.http
        )
    # Return
    return web_app

@asynccontextmanager
async def _lifespan(wapp: FastAPI):
    """
    Spin up the routing table refresh loop once the server starts.
    """
    # One shared client for probing backends and forwarding completions.
    async with AsyncClient(follow_redirects=True) as http:
        wapp.state.http = http
        wapp.state.routes = {}
        wapp.state.last_refresh = None
        # Discover in the background so readiness never blocks on the control plane.
        refresh_task = create_task(_refresh_loop(wapp))
        try:
            yield
        finally:
            refresh_task.cancel()

async def _refresh_loop(
    wapp: FastAPI,
    *,
    refresh_interval: float = 10.
):
    """
    Rebuild the `tag -> [url]` routing table on `wapp.state.routes`.

    On error the previous table is kept (fail-stale), so a control-plane hiccup never
    drops models that are still being served.
    """
    while True:
        try:
            # Get all backend URLs from our Modal workspace
            urls = await wait_for(
                to_thread(_list_backend_urls),
                timeout=30
            )
            # Gather all backends (URL + loaded models)
            backends = await gather(*(
                _probe_models(url, http=wapp.state.http)
                for url in urls
            ))
            # Gather routing table
            tags = { tag for backend in backends for tag in backend.tags }
            routes = {
                tag: [backend.url for backend in backends if tag in backend.tags]
                for tag in tags
            }
            wapp.state.routes = routes
            wapp.state.last_refresh = monotonic()
            print(f"[discovery] {len(routes)} model(s) across {len(urls)} live backend(s)")
        except Exception as error:
            print(f"[discovery] refresh failed: {error!r}")
        await sleep(refresh_interval)

async def _probe_models(
    url: str,
    *,
    http: AsyncClient,
    timeout: float=5.
) -> _Backend:
    """
    Probe a backend for its loaded models.
    """
    try: # backends can scale down at any time; treat unreachable as serving nothing
        response = await http.get(f"{url}/v1/models", timeout=timeout)
        response.raise_for_status()
    except HTTPError:
        return _Backend(url=url)
    try:
        models = Page[Model].model_validate_json(response.content)
    except ValidationError: # not a muna-server (e.g. muna-qnn / muna-rpc)
        return _Backend(url=url)
    return _Backend(url=url, tags=[model.id for model in models.data])

@synchronizer.create_blocking
async def _list_backend_urls(environment: str = "main") -> list[str]:
    """
    List the web endpoint URLs of every live `muna-*` app in the workspace.

    We read URLs from the app layout's function metadata rather than looking functions up
    by name — the muna-server web function is a nested local, so its registered name is
    mangled and version-dependent. Any function exposing a `web_url` is a candidate.

    Modal SDK calls must run on Modal's own `synchronizer` event loop, not uvicorn's, or the
    gRPC stub calls hang ("RPC ... made outside of task context"). `create_blocking` returns
    a plain blocking function that dispatches onto that loop; the caller runs it in a worker
    thread (`asyncio.to_thread`) so uvicorn's loop is never blocked.
    """
    client = await _Client.from_env()
    apps = await client.stub.AppList(api_pb2.AppListRequest(environment_name=environment))
    candidates = [
        deployment.app_id for deployment in apps.apps
        if (deployment.name or "").startswith("muna-")
        and deployment.name != app.name        # never probe ourselves
        and deployment.n_running_tasks > 0     # skip idle/scaled-to-zero apps
    ]
    urls: list[str] = []
    for app_id in candidates:
        try:
            layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
        except Exception:
            continue
        for obj in layout.app_layout.objects:
            if url := obj.function_handle_metadata.web_url:
                urls.append(url.rstrip("/"))
    return urls

async def _forward(
    url: str,
    body: dict,
    *,
    http: AsyncClient,
    timeout: Timeout | None = None
) -> StreamingResponse | JSONResponse:
    """
    Forward a request to a backend, relaying its response verbatim.

    Streaming the response body handles both cases uniformly: SSE completions are relayed
    chunk by chunk, plain JSON bodies arrive as a single chunk. Backend errors (non-200)
    pass through untouched; only transport failures produce our own 502.
    """
    # Bound connect/write but never the read — generations stream for a long time, and a
    # just-scaled-down backend may need a moment to accept the connection.
    DEFAULT_TIMEOUT = Timeout(connect=30., read=None, write=30., pool=30.)
    timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
    request = http.build_request("POST", url, json=body, timeout=timeout)
    try:
        response = await http.send(request, stream=True)
    except HTTPError as error:
        return _error(
            f"Backend request failed: {error}",
            status=502,
            code="backend_error"
        )
    async def stream():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
    return StreamingResponse(
        stream(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type"),
    )

def _error(
    message: str,
    *,
    status: int,
    code: str
) -> JSONResponse:
    """
    Render an OpenAI-style error payload as an HTTP response.
    """
    response = ErrorResponse(error=ErrorObject(message=message, code=code))
    return JSONResponse(status_code=status, content=response.model_dump())

def _render_landing(routes: dict[str, list[str]]) -> str:
    """
    Render the landing page with the current routing table.
    """
    tags = sorted(routes)
    chips = "\n".join(
        f'<span class="chip chip-{index % 3}">{escape(tag)}</span>'
        for index, tag in enumerate(tags)
    ) or '<span class="empty">Warming up — discovering live models…</span>'
    return _LANDING_PAGE.substitute(
        model_chips=chips,
        example_model=escape(tags[0]) if tags else "@openai/gpt-oss-20b",
    )

T = TypeVar("T")

class Page(BaseModel, Generic[T]):
    """
    Mirrors the OpenAI SDK's `SyncPage[T]` list type.
    """
    object: Literal["list"] = Field("list", init=False)
    data: list[T] = Field(default_factory=list)

class Model(BaseModel):
    """
    Loaded model.
    """
    id: str
    object: Literal["model"] = Field("model", init=False)
    created: int = Field(default_factory=lambda: int(time()))
    owned_by: str = "muna"

class ErrorObject(BaseModel):
    """
    Mirrors the OpenAI SDK's `ErrorObject` type.
    """
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None

class ErrorResponse(BaseModel):
    """
    The envelope OpenAI error payloads are wrapped in on the wire.
    """
    error: ErrorObject

class _Backend(BaseModel):
    """
    A live backend and the model tags it currently serves.
    """
    url: str
    tags: list[str] = []

# Design tokens (colors, fonts, dashed hairlines, pill chips) are lifted from
# muna.ai so the gateway feels like part of the same product.
_LANDING_PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Muna Inference — OpenAI-compatible endpoint</title>
<link rel="icon" type="image/png" href="https://www.muna.ai/logo_1024.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/vs2015.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
<style>
:root {
    --foreground: 210 20% 98%;
    --muted: 217.9 10.6% 64.9%;
    --hairline: rgb(229 231 235 / .2);
}
* { box-sizing: border-box; }
html { color-scheme: dark; }
body {
    margin: 0;
    background: #000;
    color: hsl(var(--foreground));
    font-family: "Geist", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}
code, pre, .chip, .wordmark, footer {
    font-family: "Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
main { max-width: 42rem; margin: 0 auto; padding: 3rem 1.5rem 4rem; }
header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 2.5rem; }
.brand { display: flex; align-items: center; gap: .625rem; }
.logo { width: 2.5rem; height: 2.5rem; }
.wordmark { font-weight: 500; letter-spacing: .02em; }
.nav { display: flex; gap: 1.25rem; }
.nav-link { color: hsl(var(--muted)); text-decoration: none; font-size: .875rem; }
.nav-link:hover { color: hsl(var(--foreground)); }
h1 { font-size: 2.25rem; line-height: 1.15; font-weight: 600; letter-spacing: -.02em; margin: 0 0 1rem; }
h2 { font-size: 1.125rem; font-weight: 600; letter-spacing: -.01em; margin: 0 0 1.25rem; }
.lede { color: hsl(var(--muted)); line-height: 1.6; margin: 0 0 1.5rem; }
.lede code { font-size: .875em; color: hsl(var(--foreground)); }
section { padding: 2rem 0; }
section + section, footer { border-top: 1px dashed var(--hairline); }
.hero { padding-top: 0; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip { display: inline-flex; align-items: center; height: 1.75rem; padding: 0 .75rem; border-radius: 9999px; border: 1px solid; font-size: .75rem; }
.chip-0 { border-color: rgb(52 211 153 / .5); color: #a7f3d0; background: rgb(16 185 129 / .06); }
.chip-1 { border-color: rgb(129 140 248 / .5); color: #c7d2fe; background: rgb(99 102 241 / .06); }
.chip-2 { border-color: rgb(244 114 182 / .5); color: #fbcfe8; background: rgb(244 114 182 / .06); }
.empty { color: hsl(var(--muted)); font-size: .875rem; }
pre { margin: 0 0 1.5rem; padding: 1.25rem; border: 1px dashed var(--hairline); border-radius: .5rem; background: rgb(255 255 255 / .03); overflow-x: auto; font-size: .8125rem; line-height: 1.6; }
/* Keep our own pre chrome; the hljs theme only provides token colors. */
pre code.hljs { background: transparent; padding: 0; }
.cta { display: inline-flex; align-items: center; height: 2.75rem; padding: 0 1.5rem; border-radius: 9999px; background: hsl(210 20% 98%); color: hsl(220.9 39.3% 11%); font-weight: 500; font-size: .9375rem; text-decoration: none; }
.cta:hover { background: hsl(210 20% 90%); }
footer { display: flex; gap: .75rem; padding-top: 2rem; font-size: .8125rem; color: hsl(var(--muted)); }
footer a { color: hsl(var(--muted)); text-decoration: none; }
footer a:hover { color: hsl(var(--foreground)); }
.spacer { flex: 1; }
</style>
</head>
<body>
<main>
    <header>
        <span class="brand">
            <img class="logo" src="https://www.muna.ai/logo_1024.png" alt="Muna logo">
            <span class="wordmark">muna</span>
        </span>
        <nav class="nav">
            <a class="nav-link" href="https://docs.muna.ai">docs</a>
            <a class="nav-link" href="https://muna.ai">muna.ai →</a>
        </nav>
    </header>
    <section class="hero">
        <h1>Compiled inference.<br>Your OpenAI client.</h1>
        <p class="lede">
            This is Muna's OpenAI-compatible inference gateway. Point the official
            OpenAI SDK at <code>https://inference.muna.ai/v1</code> and call any
            compiled model below.
        </p>
    </section>
    <section>
        <h2>Live models</h2>
        <div class="chips">
            $model_chips
        </div>
    </section>
    <section>
        <h2>Quick start</h2>
        <pre><code class="language-python">from openai import OpenAI

# 💥 Create a client with the Muna URL
client = OpenAI(
    base_url="https://inference.muna.ai/v1",
    api_key="&lt;your Muna API key&gt;"
)

# 🔥 Create a chat completion
completion = client.chat.completions.create(
    model="$example_model",
    messages=[{ "role": "user", "content": "What is a GPU?" }]
)

# 🚀 Print the output
print(completion.choices[0].message.content)</code></pre>
        <a class="cta" href="https://muna.ai">Get an API key</a>
    </section>
    <footer>
        <a href="https://muna.ai">muna.ai</a>
        <span>·</span>
        <a href="/v1/models">/v1/models</a>
        <span>·</span>
        <a href="/health">/health</a>
        <span class="spacer"></span>
        <span>© 2026 NatML Inc.</span>
    </footer>
</main>
<script>hljs.highlightAll();</script>
</body>
</html>""")