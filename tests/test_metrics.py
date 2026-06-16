def _chat(app_client, content: str) -> None:
    app_client.post(
        "/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        },
    )


def test_metrics_json_shape_after_requests(app_client) -> None:
    for i in range(10):
        if i % 2 == 0:
            _chat(app_client, "What model do you use?")
        else:
            _chat(app_client, f"Novel question number {i} about zorbplix")

    response = app_client.get("/metrics.json")
    assert response.status_code == 200
    payload = response.json()

    expected_keys = {
        "p50_ms",
        "p95_ms",
        "tok_per_sec",
        "cache_hit_rate",
        "requests_total",
        "cost_per_1k_estimate",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["requests_total"] == 10
    assert payload["cache_hit_rate"] > 0
    assert isinstance(payload["p50_ms"], (int, float))
    assert isinstance(payload["p95_ms"], (int, float))