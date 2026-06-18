from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.cache import SemanticCache
from app.chat import extract_prompt, handle_chat_completions
from app.config import Settings, load_settings
from app.embeddings import get_embedder, reset_embedder
from app.request_log import RequestLogger
from app.safety import classify
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
    if http_client is None or embedder is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Pre-LLM safety filter — ported from rhobear-sales-chat/src/safety.py.
    # Runs before retrieval/cache/LLM so the upstream model never sees the
    # message and no lead capture is offered. Shape mirrors the rest of the
    # chat completions response so downstream consumers (sales bubble, ops
    # tooling) don't need to special-case the safety refusal.
    user_msg = extract_prompt(body.get("messages", []))
    decision = classify(user_msg)
    if decision.category:
        return JSONResponse(
            {
                "id": f"chatcmpl-safety-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "rhobear-safety-refusal",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": decision.response or "",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

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
    if http_client is None:
        return JSONResponse(
            {
                "status": "degraded",
                "cache": cache_ok,
                "requests_log": requests_ok,
                "llama_upstream": False,
            },
            status_code=503,
        )
    try:
        response = await http_client.get(
            f"{settings.llama_upstream.rstrip('/')}/health",
            timeout=2.0,
        )
        upstream_ok = 200 <= response.status_code < 300
    except httpx.HTTPError:
        try:
            response = await http_client.get(settings.llama_upstream, timeout=2.0)
            upstream_ok = 200 <= response.status_code < 300
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

    raw = await request.body()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Body is not valid UTF-8: {exc}") from exc

    pairs = []
    errors: list[dict[str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            question = data["q"]
            answer = data["a"]
        except json.JSONDecodeError as exc:
            errors.append({"line": str(line_no), "error": f"invalid JSON: {exc.msg}"})
            continue
        except KeyError as exc:
            errors.append({"line": str(line_no), "error": f"missing key: {exc.args[0]}"})
            continue
        from app.seed import SeedPair

        pairs.append(
            SeedPair(
                question=question,
                answer=answer,
                thought=data.get("thought"),
            )
        )

    if errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "Malformed seed lines", "errors": errors},
        )

    if embedder is None:
        raise HTTPException(status_code=503, detail="Service not ready")

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