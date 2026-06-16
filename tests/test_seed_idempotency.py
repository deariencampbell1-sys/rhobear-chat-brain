def test_seed_loader_is_idempotent(app_client) -> None:
    import app.main as main_module

    count_after_boot = main_module.cache.count_entries()
    assert count_after_boot == 30

    response = app_client.post(
        "/admin/seed",
        headers={"X-Admin-Token": "test-admin-token"},
        content='{"q": "How does this chatbot work?", "a": "duplicate"}\n',
    )
    assert response.status_code == 200
    assert response.json()["inserted"] == 0
    assert response.json()["skipped"] == 1
    assert main_module.cache.count_entries() == 30