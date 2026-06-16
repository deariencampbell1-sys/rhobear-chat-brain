import time

import pytest


@pytest.mark.parametrize(
    "question",
    [
        "What model do you use?",
        "How does this chatbot work?",
    ],
)
def test_cache_hit_returns_cached_answer(app_client, question: str) -> None:
    started = time.perf_counter()
    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "rhobear-cache"
    assert payload["choices"][0]["message"]["content"]
    assert elapsed_ms < 2000


def test_cache_hit_under_200ms_for_seeded_model_question(app_client) -> None:
    started = time.perf_counter()
    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "What model do you use?"}],
            "stream": False,
        },
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "llama.cpp" in content.lower()
    assert elapsed_ms < 200