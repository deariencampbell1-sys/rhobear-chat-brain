from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.cache import CacheMatch, SemanticCache
from app.embeddings import Embedder
from app.providers import (
    BaseProvider,
    ChainProvider,
    LlamaProvider,
    MiniMaxProvider,
    build_chain,
    classify_task,
)
from app.request_log import RequestLogger, RequestRecord

TOKENS_PER_SECOND = 40
TOKEN_INTERVAL_SEC = 1.0 / TOKENS_PER_SECOND


def extract_prompt(messages: list[Any]) -> str:
    if not isinstance(messages, list):
        return ""
    user_messages: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            user_messages.append(str(m.get("content", "")))
    if not user_messages:
        return ""
    return user_messages[-1].strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_answer(match: CacheMatch) -> str:
    if match.thought:
        return f"{match.thought}\n\n{match.answer}"
    return match.answer


def tokenize_for_stream(text: str) -> list[str]:
    tokens: list[str] = []
    for word in text.split(" "):
        if not tokens:
            tokens.append(word)
        else:
            tokens.append(" " + word)
    if not tokens and text:
        tokens.append(text)
    return tokens


def make_chunk(content: str, *, finish_reason: str | None = None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "rhobear-cache",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content.split()),
            "total_tokens": len(content.split()),
        },
    }


async def stream_cached_answer(answer: str) -> AsyncIterator[str]:
    for token in tokenize_for_stream(answer):
        yield make_chunk(token)
        await _async_sleep(TOKEN_INTERVAL_SEC)
    yield make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def build_providers(
    *,
    minimax_api_key: str,
    minimax_base_url: str,
    minimax_fast_model: str,
    minimax_thinker_model: str,
    llama_upstream: str,
) -> tuple[MiniMaxProvider, MiniMaxProvider, LlamaProvider]:
    """Construct the three provider instances we use in the chain."""
    fast = MiniMaxProvider(minimax_fast_model, minimax_api_key, minimax_base_url)
    thinker = MiniMaxProvider(minimax_thinker_model, minimax_api_key, minimax_base_url)
    local = LlamaProvider(llama_upstream)
    return fast, thinker, local


async def handle_chat_completions(
    request: Request,
    body: dict[str, Any],
    *,
    cache: SemanticCache,
    embedder: Embedder,
    request_logger: RequestLogger,
    cache_threshold: float,
    http_client: httpx.AsyncClient,
    fast_provider: BaseProvider,
    thinker_provider: BaseProvider,
    local_provider: BaseProvider,
) -> JSONResponse | StreamingResponse:
    started = time.perf_counter()
    messages = body.get("messages", [])
    prompt = extract_prompt(messages)
    p_hash = prompt_hash(prompt)
    stream = bool(body.get("stream", False))
    model = str(body.get("model", "rhobear-chat-brain"))

    embedding = embedder.embed(prompt)
    match = cache.search(embedding, cache_threshold)

    if match is not None:
        answer = build_answer(match)
        elapsed_ms = (time.perf_counter() - started) * 1000
        tokens_out = max(1, len(answer.split()))

        request_logger.log(
            RequestRecord(
                ts=time.time(),
                prompt_hash=p_hash,
                hit_or_miss="hit",
                similarity_score=match.similarity,
                response_ms=elapsed_ms,
                tokens_out=tokens_out,
                model_used="rhobear-cache",
            )
        )

        if stream:
            return StreamingResponse(
                stream_cached_answer(answer),
                media_type="text/event-stream",
            )

        return JSONResponse(make_completion_response(answer, "rhobear-cache"))

    # Cache miss: classify task, build provider chain, call.
    task = classify_task(prompt)
    chain = build_chain(
        task,
        fast_provider=fast_provider,
        thinker_provider=thinker_provider,
        local_provider=local_provider,
    )

    async def on_miss_complete(
        response_text: str,
        upstream_model: str,
        tokens_out: int | None = None,
        provider_name: str = "",
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000
        if tokens_out is None:
            tokens_out = max(1, len(response_text.split())) if response_text else 1
        request_logger.log(
            RequestRecord(
                ts=time.time(),
                prompt_hash=p_hash,
                hit_or_miss="miss",
                similarity_score=None,
                response_ms=elapsed_ms,
                tokens_out=tokens_out,
                model_used=upstream_model or provider_name,
            )
        )

    if stream:
        async def stream_chain() -> AsyncIterator[bytes]:
            upstream_model = model
            tokens_out = 0
            line_buffer = ""
            provider_served = ""
            # For streaming we only try the first provider in the chain.
            # (Fallback after streaming has started is not feasible.)
            first = chain.providers[0]
            try:
                async for chunk in first.stream(http_client, body):
                    tokens_out += max(1, len(chunk) // 4)
                    if upstream_model == model:
                        line_buffer += chunk.decode("utf-8", errors="replace")
                        while "\n" in line_buffer:
                            head, _, line_buffer = line_buffer.partition("\n")
                            head = head.strip()
                            if head.startswith("data: ") and head != "data: [DONE]":
                                try:
                                    payload = json.loads(head[6:])
                                    upstream_model = payload.get("model", upstream_model)
                                except json.JSONDecodeError:
                                    pass
                                break
                    yield chunk
                provider_served = first.name
            except Exception as e:  # noqa: BLE001
                # First provider failed mid-stream: synthesize an error chunk
                err = {
                    "error": {
                        "message": f"primary provider failed: {e}",
                        "type": "upstream_error",
                    }
                }
                yield f"data: {json.dumps(err)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            await on_miss_complete(
                "", upstream_model, tokens_out=max(1, tokens_out), provider_name=provider_served,
            )

        return StreamingResponse(stream_chain(), media_type="text/event-stream")

    # Non-streaming: full chain with fallback
    try:
        result = await chain.chat(http_client, body, timeout=60.0)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": {"message": f"all providers failed: {e}", "type": "upstream_error"}},
            status_code=502,
        )
    await on_miss_complete(result.content, result.model, provider_name=result.provider)
    if result.raw is not None:
        # Preserve the upstream response shape (id, created, usage, etc.) but
        # surface the model + provider that actually answered.
        result.raw["model"] = result.model
        result.raw.setdefault("rhobear_provider", result.provider)
        return JSONResponse(result.raw)
    return JSONResponse(make_completion_response(result.content, result.model))
