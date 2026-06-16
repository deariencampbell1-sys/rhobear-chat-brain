from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.cache import SemanticCache
from app.chat import handle_chat_completions
from app.config import Settings, load_settings
from app.embeddings import get_embedder, reset_embedder
from app.request_log import RequestLogger
from app.seed import load_seed_file, seed_cache

settings: Settings = load_settings()
cache = SemanticCache(settings.cache_db_path, settings.embedding_dim)
request_logger = RequestLogger(settings.requests_db_path)
http_client: httpx.AsyncClient | None = None
embedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, embedder

    test_mode = settings.admin_token == "test-admin-token"
    cache.connect()
    request_logger.connect()
    embedder = get_embedder(
        settings.embedding_model,
        test_mode=test_mode,
        dim=settings.embedding_dim,
        onnx_model_dir=settings.onnx_model_dir,
    )

    pairs = load_seed_file(settings.seeds_path)
    if pairs:
        seed_cache(cache, embedder, pairs)

    http_client = httpx.AsyncClient()
    yield
    await http_client.aclose()
    cache.close()
    request_logger.close()
    reset_embedder()


app = FastAPI(title="rhobear-chat-brain", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    assert http_client is not None and embedder is not None
    return await handle_chat_completions(
        request,
        body,
        cache=cache,
        embedder=embedder,
        request_logger=request_logger,
        upstream_url=settings.chat_completions_url,
        cache_threshold=settings.cache_threshold,
        http_client=http_client,
    )


@app.get("/metrics.json")
async def metrics() -> dict[str, Any]:
    return request_logger.metrics()


@app.get("/healthz")
async def healthz() -> JSONResponse:
    cache_ok = cache.is_reachable()
    requests_ok = request_logger.is_reachable()
    upstream_ok = False
    assert http_client is not None
    try:
        response = await http_client.get(
            f"{settings.llama_upstream.rstrip('/')}/health",
            timeout=2.0,
        )
        upstream_ok = response.status_code < 500
    except httpx.HTTPError:
        try:
            response = await http_client.get(settings.llama_upstream, timeout=2.0)
            upstream_ok = response.status_code < 500
        except httpx.HTTPError:
            upstream_ok = False

    if cache_ok and requests_ok and upstream_ok:
        return JSONResponse({"status": "ok"}, status_code=200)
    return JSONResponse(
        {
            "status": "degraded",
            "cache": cache_ok,
            "requests_log": requests_ok,
            "llama_upstream": upstream_ok,
        },
        status_code=503,
    )


@app.post("/admin/seed")
async def admin_seed(
    request: Request,
    x_admin_token: str | None = Header(default=None),
) -> dict[str, Any]:
    if not settings.admin_token:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    body = await request.body()
    pairs = []
    for line in body.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        from app.seed import SeedPair

        pairs.append(
            SeedPair(
                question=data["q"],
                answer=data["a"],
                thought=data.get("thought"),
            )
        )

    assert embedder is not None
    result = seed_cache(cache, embedder, pairs)
    return {"status": "ok", **result}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()