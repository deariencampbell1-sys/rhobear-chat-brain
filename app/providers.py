"""LLM provider abstraction with chain-of-fallback.
Primary chain: MiniMax M2.7-highspeed (fast) for casual lookups,
               MiniMax M3 (thinker) for code/email/reasoning.
Last resort:   local llama.cpp server.

MiniMax wire format (matches Apollo + Hermes canon):
  POST {base_url}/v1/text/chatcompletion_v2
  Headers: Authorization: Bearer <key>, Content-Type: application/json
  Body:    {"model": ..., "messages": [...], "max_tokens": ..., "stream": true}
  Stream:  SSE — one JSON object per line, each with
           choices[0].delta.content (or .reasoning_content) and a
           terminal "data: [DONE]".

The base URL is the project root (no trailing /v1) — the path is
appended by this provider. This matches the env keys used by Apollo
(MINIMAX_BASE_URL=https://api.minimax.io) and Hermes.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, AsyncIterator
import httpx

# --- Langfuse tracing (best-effort, never raises) ---
try:
    from langfuse import Langfuse
    _lf = Langfuse() if os.environ.get("LANGFUSE_PUBLIC_KEY") else None
except Exception as _lf_init_err:
    _lf = None


def _trace_generation(name, model, input_msgs, output_text, metadata=None):
    """Emit a langfuse generation if the client is configured; swallow errors."""
    if _lf is None:
        return
    try:
        with _lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_msgs,
            output=output_text,
            metadata=metadata or {},
        ) as gen:
            try:
                gen.update(output=output_text)
            except Exception:
                pass
    except Exception:
        # Tracing must never break the provider chain.
        pass


@dataclass
class ProviderResult:
    content: str
    model: str
    raw: dict[str, Any] | None = None
    provider: str = ""


class BaseProvider:
    name: str = "base"

    async def chat(self, client: httpx.AsyncClient, body: dict, timeout: float = 60.0) -> ProviderResult:
        raise NotImplementedError

    async def stream(self, client: httpx.AsyncClient, body: dict) -> AsyncIterator[bytes]:
        raise NotImplementedError
        yield b""  # pragma: no cover


# MiniMax canonical endpoint path. Matches Apollo llm_review.py and
# Hermes llm.py. The base URL stored in settings is the project root
# (no /v1 suffix); this constant is what we append.
_MINIMAX_CHAT_PATH = "/v1/text/chatcompletion_v2"


class MiniMaxProvider(BaseProvider):
    """MiniMax direct-HTTP provider (same shape as Apollo + Hermes).

    Wire format: SSE on the streaming path, single JSON object on the
    non-streaming path. See module docstring.
    """

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.minimax.io"):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.name = f"minimax:{model}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self) -> str:
        return f"{self.base_url}{_MINIMAX_CHAT_PATH}"

    async def chat(self, client, body, timeout=60.0):
        # Non-streaming call. The gateway passes the body through,
        # including any stream flag the client set. We respect the
        # caller's preference: if they asked for streaming, we get
        # an SSE body; otherwise we expect a single JSON object.
        # Both formats are handled here.
        payload = {**body, "model": self.model}
        if "stream" not in payload:
            payload["stream"] = False
        r = await client.post(self._url(), json=payload, headers=self._headers(), timeout=timeout)
        r.raise_for_status()
        text = r.text
        # First try a single-JSON response (the OpenAI-compatible
        # shape, also what some MiniMax endpoints return when
        # stream=False).
        try:
            data = r.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and data.get("choices"):
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            _trace_generation(
                name="chat-brain.minimax.chat",
                model=self.model,
                input_msgs=payload.get("messages"),
                output_text=content,
                metadata={"provider": self.name, "stream": False},
            )
            return ProviderResult(content=content, model=self.model, raw=data, provider=self.name)
        # Fall back to SSE assembly.
        content, _chunks = _assemble_sse(text)
        raw = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
        _trace_generation(
            name="chat-brain.minimax.chat",
            model=self.model,
            input_msgs=payload.get("messages"),
            output_text=content,
            metadata={"provider": self.name, "stream": False, "fallback": "sse"},
        )
        return ProviderResult(content=content, model=self.model, raw=raw, provider=self.name)

    async def stream(self, client, body):
        payload = {**body, "model": self.model, "stream": True}
        accumulated = []
        try:
            async with client.stream("POST", self._url(), json=payload, headers=self._headers(), timeout=60.0) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if chunk:
                        accumulated.append(chunk)
                    yield chunk
        finally:
            # Best-effort tracing of the assembled stream content.
            try:
                blob = b"".join(accumulated).decode("utf-8", errors="replace")
                content, _ = _assemble_sse(blob)
                _trace_generation(
                    name="chat-brain.minimax.stream",
                    model=self.model,
                    input_msgs=payload.get("messages"),
                    output_text=content,
                    metadata={"provider": self.name, "stream": True},
                )
            except Exception:
                pass


def _assemble_sse(body: str) -> tuple[str, int]:
    """Parse an SSE response body and return ``(content, chunk_count)``.

    Mirrors the parser in hermes/src/llm.py: one JSON object per
    ``data:`` line; each may carry ``choices[0].delta.content``
    or ``choices[0].delta.reasoning_content``; the terminal
    ``data: [DONE]`` ends the stream.
    """
    import json as _json
    pieces: list[str] = []
    chunks = 0
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = _json.loads(data)
        except _json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "error" in obj and not obj.get("choices"):
            # Surface upstream error in the assembled content so the
            # gateway's request log records the reason.
            return (f"[minimax error: {obj['error']}]", chunks)
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            pieces.append(piece)
            chunks += 1
    return "".join(pieces), chunks


class LlamaProvider(BaseProvider):
    """Local llama.cpp server (OpenAI-compatible)."""

    def __init__(self, upstream: str):
        self.upstream = upstream.rstrip("/")
        self.name = f"llama:{self.upstream}"

    async def chat(self, client, body, timeout=120.0):
        url = f"{self.upstream}/v1/chat/completions"
        r = await client.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        _trace_generation(
            name="chat-brain.llama.chat",
            model=data.get("model", "llama"),
            input_msgs=body.get("messages"),
            output_text=content,
            metadata={"provider": self.name, "stream": False},
        )
        return ProviderResult(content=content, model=data.get("model", "llama"), raw=data, provider=self.name)

    async def stream(self, client, body):
        url = f"{self.upstream}/v1/chat/completions"
        accumulated = []
        try:
            async with client.stream("POST", url, json=body, timeout=120.0) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if chunk:
                        accumulated.append(chunk)
                    yield chunk
        finally:
            try:
                blob = b"".join(accumulated).decode("utf-8", errors="replace")
                content, _ = _assemble_sse(blob)
                _trace_generation(
                    name="chat-brain.llama.stream",
                    model=body.get("model", "llama"),
                    input_msgs=body.get("messages"),
                    output_text=content,
                    metadata={"provider": self.name, "stream": True},
                )
            except Exception:
                pass


class ChainProvider(BaseProvider):
    """Tries providers in order; first success wins."""

    def __init__(self, providers: list[BaseProvider]):
        self.providers = providers
        self.name = "chain:" + ",".join(p.name for p in providers)

    async def chat(self, client, body, timeout=60.0):
        last_err: Exception | None = None
        for p in self.providers:
            try:
                return await p.chat(client, body, timeout)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"all providers failed: {last_err}")

    async def stream(self, client, body):
        last_err: Exception | None = None
        for p in self.providers:
            try:
                async for chunk in p.stream(client, body):
                    yield chunk
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"all stream providers failed: {last_err}")


# ---------------------------------------------------------------------------
# Task classifier
# ---------------------------------------------------------------------------

HEAVY_KEYWORDS = (
    "code", "implement", "function", "class ", "method ", "script", "regex",
    "compile", "build ", "package ", "module ", "import ", "syntax",
    "fix", "bug", "debug", "traceback", "exception", "stack trace", "refactor",
    "patch", "diff ", "commit", "merge", "rebase", "pull request", "code review",
    "sql ", "query ", "schema", "migration", "dockerfile", "yaml", "toml",
    "test", "unit test", "pytest", "jest",
    "email", "compose", "draft", "reply to", "subject line", "tone", "outreach",
    "analyze", "analyse", "reason", "reasoning", "step by step", "step-by-step",
    "plan", "strategy", "compare", "evaluate", "trade-off", "tradeoff",
    "explain why", "checklist", "to-do", "todo list", "task list", "list of ",
    "walk me through", "walkthrough",
)

FAST_KEYWORDS = (
    "look up", "find ", "search ", "what is", "where is", "who is",
    "go check", "hit the repo", "browse", "fetch", "url", "link",
    "summary", "summarize", "summarise", "tldr", "tl;dr", "short version",
    "quick ", "brief ", "1-line", "one-line", "just tell me",
)


def classify_task(prompt: str) -> str:
    """Return 'heavy' or 'fast' based on keyword heuristics."""
    p = prompt.lower()
    heavy = sum(1 for kw in HEAVY_KEYWORDS if kw in p)
    fast = sum(1 for kw in FAST_KEYWORDS if kw in p)
    return "heavy" if heavy > fast else "fast"


def build_chain(
    task: str,
    *,
    fast_provider: BaseProvider,
    thinker_provider: BaseProvider,
    local_provider: BaseProvider,
) -> ChainProvider:
    """Build the provider chain for a given task.

    Primary is task-dependent (thinker for heavy, fast for casual). If primary
    fails, fall through to the other cloud model, then to local llama.cpp.
    """
    # Skip MiniMax providers whose key is empty — otherwise the chain
    # would attempt to call the cloud with an empty bearer and either
    # round-trip the real internet (tests) or fail every first attempt
    # in prod (any moment the key is briefly unset). Local llama always
    # stays — it's the safety net.
    def _ok(p: BaseProvider) -> bool:
        return getattr(p, "api_key", "x") != ""

    if task == "heavy":
        order = [thinker_provider, fast_provider, local_provider]
    else:
        order = [fast_provider, thinker_provider, local_provider]
    order = [p for p in order if _ok(p)]
    if not order:
        order = [local_provider]
    return ChainProvider(order)
