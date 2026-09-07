import pytest
from hte_agent.offline import offline_execution, deny_network
import urllib.request


@pytest.fixture(autouse=True)
def network_free_tests(monkeypatch):
    with offline_execution():
        monkeypatch.setattr(urllib.request, "urlopen", deny_network)
        yield
