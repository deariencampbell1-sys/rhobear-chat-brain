"""Smoke test: the langfuse wrapper is reachable without env vars.

The chat-brain provider chain must no-op cleanly when LANGFUSE_PUBLIC_KEY
is unset (e.g. local CI, dev, tests). This guards the no-env graceful
path so a regression that raises on import is caught early.
"""
from __future__ import annotations

import importlib

import pytest


def test_langfuse_module_loads_without_key(monkeypatch):
    """Importing app.providers without LANGFUSE_PUBLIC_KEY must succeed
    and the wrapper must be a safe no-op (no exception, no network)."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    # Force a fresh import so the env-read happens under our monkeypatch.
    import app.providers as providers
    importlib.reload(providers)
    # Must not raise and must not require network.
    providers._trace_generation(
        name="smoke.noop",
        model="noop-model",
        input_msgs=[{"role": "user", "content": "hi"}],
        output_text="hello back",
        metadata={"smoke": True},
    )


def test_langfuse_wrapper_noop_with_unset_key():
    """Calling _trace_generation without the env var must be a clean no-op
    (return None, no exception)."""
    import os
    import app.providers as providers

    saved_pub = os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    try:
        # The module may have been loaded with the key set; this test
        # asserts the no-env branch behaves correctly regardless.
        result = providers._trace_generation(
            name="smoke.noop2",
            model="noop-model",
            input_msgs=None,
            output_text="",
            metadata={},
        )
        assert result is None
    finally:
        if saved_pub is not None:
            os.environ["LANGFUSE_PUBLIC_KEY"] = saved_pub