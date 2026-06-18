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
from app.chat import build_providers, extract_prompt, handle_chat_completions
from app.config import Settings, load_settings
from app.embeddings import get_embedder, reset_embedder
from app.pi_session import get_manager
from app.request_log import RequestLogger
from app.safety import classify
from app.seed import load_seed_file, seed_cache

settings: Settings = load_settings()
cache = SemanticCache(settings.cache_db_path, settings.embedding_dim)
request_logger = RequestLogger(settings.requests_db_path)
http_client: httpx.AsyncClient | None = None
embedder = None

# Provider chain (built once at import; uses settings from env / systemd)
fast_provider, thinker_provider, local_provider = build_providers(
    minimax_api_key=settings.minimax_api_key,
    minimax_base_url=settings.minimax_base_url,
    minimax_fast_model=settings.minimax_fast_model,
    minimax_thinker_model=settings.minimax_thinker_model,
    llama_upstream=settings.llama_upstream,
)


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
    # Bring up the long-running pi session manager so Telegram / build
    # requests can hit a persistent pi process per chat.
    pi_mgr = get_manager()
    try:
        yield
    finally:
        pi_mgr.shutdown()
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
        cache_threshold=settings.cache_threshold,
        http_client=http_client,
        fast_provider=fast_provider,
        thinker_provider=thinker_provider,
        local_provider=local_provider,
    )


@app.get("/metrics.json")
async def metrics() -> dict[str, Any]:
    return request_logger.metrics()


# --- Pi build sessions --------------------------------------------------
# Telegram → /v1/pi/chat → long-running `pi --mode rpc` per chat_id.
# The thread never closes on its own; /new, /start, /wipe reset it.
# See app/pi_session.py for the manager and app.pi_session.WIPE_COMMANDS
# for the wipe vocabulary.

from pydantic import BaseModel, Field


class PiChatRequest(BaseModel):
    chat_id: str = Field(..., description="Telegram chat_id (or any stable per-thread key).")
    message: str = Field(..., description="The user's prompt for pi.")
    timeout_s: int = Field(default=600, ge=10, le=3600)


class PiChatResponse(BaseModel):
    ok: bool
    text: str
    tool_calls: list[dict[str, Any]] = []
    duration_s: float
    error: str | None = None
    wiped: bool = False


@app.post("/v1/pi/chat", response_model=PiChatResponse)
async def pi_chat(req: PiChatRequest) -> PiChatResponse:
    mgr = get_manager()
    if mgr.is_wipe_command(req.message):
        msg = mgr.wipe(req.chat_id)
        return PiChatResponse(ok=True, text=msg, duration_s=0.0, wiped=True)
    result = mgr.send(req.chat_id, req.message, timeout_s=req.timeout_s)
    return PiChatResponse(**result)


@app.get("/v1/pi/sessions")
async def pi_sessions() -> dict[str, Any]:
    """List live pi RPC processes (debug aid)."""
    mgr = get_manager()
    with mgr._mu:  # noqa: SLF001 — operator-only
        return {
            "active_chats": [
                {"chat_id": cid, "pid": p.p.pid, "last_used": p.last_used}
                for cid, p in mgr._procs.items()  # noqa: SLF001
            ]
        }


async def _check_minimax(client: httpx.AsyncClient, model: str) -> bool:
    """Lightweight health probe: ask MiniMax for a 1-token completion."""
    if not settings.minimax_api_key:
        return False
    try:
        # Match the MiniMaxProvider: canonical path is
        # /v1/text/chatcompletion_v2, base URL has no /v1 suffix.
        r = await client.post(
            f"{settings.minimax_base_url.rstrip('/')}/v1/text/chatcompletion_v2",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            headers={
                "Authorization": f"Bearer {settings.minimax_api_key}",
                "Content-Type": "application/json",
            },
            timeout=3.0,
        )
        return 200 <= r.status_code < 300
    except httpx.HTTPError:
        return False


async def _check_local(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get(f"{settings.llama_upstream.rstrip('/')}/health", timeout=2.0)
        return 200 <= r.status_code < 300
    except httpx.HTTPError:
        try:
            r = await client.get(settings.llama_upstream, timeout=2.0)
            return 200 <= r.status_code < 300
        except httpx.HTTPError:
            return False


@app.get("/healthz")
async def healthz() -> JSONResponse:
    cache_ok = cache.is_reachable()
    requests_ok = request_logger.is_reachable()
    if http_client is None:
        return JSONResponse(
            {
                "status": "degraded",
                "cache": cache_ok,
                "requests_log": requests_ok,
            },
            status_code=503,
        )
    minimax_fast_ok = await _check_minimax(http_client, settings.minimax_fast_model)
    minimax_thinker_ok = await _check_minimax(http_client, settings.minimax_thinker_model)
    local_ok = await _check_local(http_client)
    all_ok = cache_ok and requests_ok and (minimax_fast_ok or minimax_thinker_ok or local_ok)
    payload = {
        "status": "ok" if all_ok else "degraded",
        "cache": cache_ok,
        "requests_log": requests_ok,
        "providers": {
            f"minimax:{settings.minimax_fast_model}": minimax_fast_ok,
            f"minimax:{settings.minimax_thinker_model}": minimax_thinker_ok,
            f"llama:{settings.llama_upstream}": local_ok,
        },
        "chain_priority": {
            "fast_tasks": [
                f"minimax:{settings.minimax_fast_model}",
                f"minimax:{settings.minimax_thinker_model}",
                f"llama:{settings.llama_upstream}",
            ],
            "heavy_tasks": [
                f"minimax:{settings.minimax_thinker_model}",
                f"minimax:{settings.minimax_fast_model}",
                f"llama:{settings.llama_upstream}",
            ],
        },
    }
    return JSONResponse(payload, status_code=200 if all_ok else 503)


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
