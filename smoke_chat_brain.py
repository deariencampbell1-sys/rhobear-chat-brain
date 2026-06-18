"""Smoke test for the rhobearvps_bot (chat-brain).

Validates:
  1. Settings load with the canonical MiniMax base URL
     (https://api.minimax.io, no /v1 suffix).
  2. The MiniMaxProvider builds the correct endpoint path
     (/v1/text/chatcompletion_v2 — same as Apollo + Hermes).
  3. The task classifier routes:
       - "what is the weather" -> fast (M2.7-highspeed)
       - "implement a function to parse JSON" -> heavy (M3)
       - "write a Python script to fix this bug" -> heavy (M3)
       - "look up the price of ETH" -> fast (M2.7-highspeed)
       - "draft an email to jane about Q3" -> heavy (M3)
  4. The chain order is:
       - heavy: thinker -> fast -> local
       - fast:  fast -> thinker -> local
  5. If MINIMAX_API_KEY is set in env, hit MiniMax with a 1-token
     ping and report the HTTP status (proves the URL is reachable
     AND the key is valid end-to-end).

Run:  python smoke_chat_brain.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# If MINIMAX_API_KEY isn't in the process env (e.g. running this
# script outside of systemd), source the canonical env file the
# service uses. systemd's EnvironmentFile= already exports the
# key when the service runs, so this is a no-op there.
if not os.environ.get("MINIMAX_API_KEY"):
    for _env_path in (
        "/etc/rhobear/chat-brain.env",
        str(Path(__file__).parent / "chat-brain.env"),
    ):
        if os.path.isfile(_env_path):
            with open(_env_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line or _line.startswith("#") or "=" not in _line:
                        continue
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
            break

# Make providers.py / config.py importable. The script lives in
# /opt/chat-brain/ but the modules live in /opt/chat-brain/app/.
# Try a few likely locations so this works without PYTHONPATH.
_script_dir = Path(__file__).resolve().parent
for _candidate in (_script_dir / "app", _script_dir.parent / "app", _script_dir):
    if (_candidate / "providers.py").is_file() and (_candidate / "config.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

import providers  # noqa: E402
import config as chat_config  # noqa: E402


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    print("== rhobearvps_bot (chat-brain) smoke test ==")

    settings = chat_config.load_settings()
    failures = 0

    # 1. Base URL default is the canonical MiniMax root
    if not check(
        "MINIMAX_BASE_URL is the canonical MiniMax root",
        settings.minimax_base_url.rstrip("/") == "https://api.minimax.io",
        settings.minimax_base_url,
    ):
        failures += 1
    if not check(
        "base URL has no trailing /v1 (provider appends path)",
        not settings.minimax_base_url.rstrip("/").endswith("/v1"),
        settings.minimax_base_url,
    ):
        failures += 1

    # 2. Model names
    if not check(
        "fast model is MiniMax-M2.7-highspeed",
        settings.minimax_fast_model == "MiniMax-M2.7-highspeed",
        settings.minimax_fast_model,
    ):
        failures += 1
    if not check(
        "thinker model is MiniMax-M3",
        settings.minimax_thinker_model == "MiniMax-M3",
        settings.minimax_thinker_model,
    ):
        failures += 1

    # 3. Provider builds the right endpoint path
    p = providers.MiniMaxProvider(
        model=settings.minimax_thinker_model,
        api_key=settings.minimax_api_key or "test",
        base_url=settings.minimax_base_url,
    )
    expected_url = "https://api.minimax.io/v1/text/chatcompletion_v2"
    if not check(
        "MiniMaxProvider builds /v1/text/chatcompletion_v2",
        p._url() == expected_url,
        p._url(),
    ):
        failures += 1

    # 4. Task classifier routing
    cases = [
        ("what is the weather in Chicago", "fast"),
        ("look up the price of ETH", "fast"),
        ("summarize this article", "fast"),
        ("implement a function to parse JSON", "heavy"),
        ("write a Python script to fix this bug", "heavy"),
        ("draft an email to jane about Q3", "heavy"),
        ("compile and run this code", "heavy"),
        ("find the docs for FastAPI", "fast"),
    ]
    for prompt, expected in cases:
        got = providers.classify_task(prompt)
        if not check(
            f"classify({prompt!r}) -> {expected}",
            got == expected,
            f"got {got}",
        ):
            failures += 1

    # 5. Chain order — build the three providers directly the same
    # way chat.build_providers does, then verify the order.
    fast_p = providers.MiniMaxProvider(
        settings.minimax_fast_model, settings.minimax_api_key or "test",
        settings.minimax_base_url,
    )
    thinker_p = providers.MiniMaxProvider(
        settings.minimax_thinker_model, settings.minimax_api_key or "test",
        settings.minimax_base_url,
    )
    local_p = providers.LlamaProvider(settings.llama_upstream)
    heavy_chain = providers.build_chain(
        "heavy", fast_provider=fast_p, thinker_provider=thinker_p, local_provider=local_p
    )
    fast_chain = providers.build_chain(
        "fast", fast_provider=fast_p, thinker_provider=thinker_p, local_provider=local_p
    )
    if not check(
        "heavy chain: thinker -> fast -> local",
        [x.name for x in heavy_chain.providers] == [
            f"minimax:{settings.minimax_thinker_model}",
            f"minimax:{settings.minimax_fast_model}",
            f"llama:{settings.llama_upstream.rstrip('/')}",
        ],
    ):
        failures += 1
    if not check(
        "fast chain: fast -> thinker -> local",
        [x.name for x in fast_chain.providers] == [
            f"minimax:{settings.minimax_fast_model}",
            f"minimax:{settings.minimax_thinker_model}",
            f"llama:{settings.llama_upstream.rstrip('/')}",
        ],
    ):
        failures += 1

    # 6. End-to-end ping if a real key is set
    if settings.minimax_api_key and settings.minimax_api_key != "test":
        print()
        print("  [probe] hitting MiniMax with a 1-token ping...")
        req = urllib.request.Request(
            f"{settings.minimax_base_url.rstrip('/')}/v1/text/chatcompletion_v2",
            data=json.dumps(
                {
                    "model": settings.minimax_fast_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.minimax_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                ok = 200 <= resp.status < 300
                if not check(
                    f"minimax ping -> HTTP {resp.status}",
                    ok,
                    body[:120].replace("\n", " "),
                ):
                    failures += 1
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if not check(
                f"minimax ping -> HTTP {e.code}",
                False,
                body[:160].replace("\n", " "),
            ):
                failures += 1
        except Exception as e:
            if not check("minimax ping reachable", False, repr(e)):
                failures += 1
    else:
        print()
        print("  [skip] MINIMAX_API_KEY not set; skipping live ping.")
        print("          Set it in chat-brain.env and re-run for an end-to-end probe.")

    print()
    if failures == 0:
        print(f"== chat-brain smoke: ALL GREEN ==")
        return 0
    print(f"== chat-brain smoke: {failures} FAILURE(S) ==")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
