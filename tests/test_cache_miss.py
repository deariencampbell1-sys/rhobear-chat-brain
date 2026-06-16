def test_cache_miss_proxies_to_upstream(app_client, respx_mock) -> None:
    upstream_body = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1,
        "model": "llama-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Upstream answer about quantum foam.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    route = respx_mock.post("http://mock-llama.test/v1/chat/completions").respond(
        200,
        json=upstream_body,
    )

    response = app_client.post(
        "/v1/chat/completions",
        json={
            "model": "llama-test",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantum foam in eleven dimensions please",
                }
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert route.called
    payload = response.json()
    assert payload["model"] == "llama-test"
    assert "quantum foam" in payload["choices"][0]["message"]["content"]