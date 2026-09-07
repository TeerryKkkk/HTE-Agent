from dataclasses import asdict

from hte_agent.llm_modules.client import OpenRouterJSONClient
from hte_agent.shared.io_utils import read_api_key


def client():
    return OpenRouterJSONClient(base_url="https://example.invalid", timeout_seconds=1, model_fallbacks=())


def test_missing_environment_disables_requests(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = client().complete_json("test", "test", {})
    assert not result["invoked"]
    assert result["status"] == "missing_api_key"


def test_secret_is_read_from_environment_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "apikey.md").write_text("untrusted-local-file", encoding="utf-8")
    assert read_api_key() == ""
    monkeypatch.setenv("OPENROUTER_API_KEY", "  test-value  ")
    assert read_api_key() == "test-value"


def test_secret_is_not_stored_in_client_metadata(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-value")
    instance = client()
    assert instance.available
    assert "test-value" not in repr(instance)
    assert "test-value" not in str(asdict(instance))
