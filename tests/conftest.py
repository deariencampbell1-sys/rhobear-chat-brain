import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault("LLAMA_UPSTREAM", "http://mock-llama.test")

TEST_ROOT = Path(__file__).resolve().parent.parent


def _upstream_json_response():
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1,
        "model": "llama-mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock upstream response."},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    os.environ["DATA_DIR"] = str(path)
    return path


@pytest.fixture
def seeds_path() -> Path:
    return TEST_ROOT / "seeds" / "sales-faq.jsonl"


@pytest.fixture
def app_client(data_dir: Path, seeds_path: Path, respx_mock) -> TestClient:
    os.environ["SEEDS_PATH"] = str(seeds_path)
    os.environ["ADMIN_TOKEN"] = "test-admin-token"
    os.environ["LLAMA_UPSTREAM"] = "http://mock-llama.test"

    respx_mock.get("http://mock-llama.test/health").respond(200, json={"status": "ok"})
    respx_mock.get("http://mock-llama.test").respond(200, text="ok")
    respx_mock.post("http://mock-llama.test/v1/chat/completions").respond(
        200,
        json=_upstream_json_response(),
    )

    from app.cache import SemanticCache
    from app.config import load_settings
    from app.embeddings import reset_embedder
    from app.request_log import RequestLogger
    import app.main as main_module

    reset_embedder()
    main_module.settings = load_settings()
    main_module.cache = SemanticCache(
        main_module.settings.cache_db_path,
        main_module.settings.embedding_dim,
    )
    main_module.request_logger = RequestLogger(
        main_module.settings.requests_db_path
    )

    with TestClient(main_module.app) as client:
        yield client

    reset_embedder()