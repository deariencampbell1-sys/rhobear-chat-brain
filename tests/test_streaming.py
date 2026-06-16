import json


def test_cache_miss_sse_chunks_pass_through_unmodified(app_client, respx_mock) -> None:
    chunk1 = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    sse_body = (
        f"data: {json.dumps(chunk1)}\n\n"
        f"data: {json.dumps(chunk2)}\n\n"
        "data: [DONE]\n\n"
    )

    route = respx_mock.post("http://mock-llama.test/v1/chat/completions").respond(
        200,
        text=sse_body,
        headers={"content-type": "text/event-stream"},
    )

    with app_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "llama-test",
            "messages": [{"role": "user", "content": "Novel streaming question xyz"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert route.called
    assert body == sse_body


def test_cache_hit_sse_format(app_client) -> None:
    with app_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "What model do you use?"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = list(response.iter_lines())

    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines
    assert data_lines[-1] == "data: [DONE]"

    first_payload = json.loads(data_lines[0].removeprefix("data: "))
    assert first_payload["object"] == "chat.completion.chunk"
    assert "choices" in first_payload