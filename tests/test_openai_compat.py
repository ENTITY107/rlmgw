"""Test OpenAI API compatibility."""

import pytest
from fastapi.testclient import TestClient

from rlmgw.config import RLMgwConfig
from rlmgw.server import RLMgwServer


def test_health_endpoint():
    """Test health endpoint."""
    config = RLMgwConfig()
    server = RLMgwServer(config)
    client = TestClient(server.app)

    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_ready_endpoint():
    """Test readiness endpoint."""
    config = RLMgwConfig()
    server = RLMgwServer(config)
    client = TestClient(server.app)

    response = client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "upstream_healthy" in data
    assert "upstream_model" in data


def test_chat_completions_streaming_rejected():
    """Test that streaming requests are rejected."""
    config = RLMgwConfig()
    server = RLMgwServer(config)
    client = TestClient(server.app)

    request_data = {
        "model": "minimax-m2-1",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

    response = client.post("/v1/chat/completions", json=request_data)
    assert response.status_code == 400
    data = response.json()
    assert "Streaming is not supported" in data["detail"]


def test_chat_completions_basic():
    """Test basic chat completions request structure."""
    config = RLMgwConfig()
    server = RLMgwServer(config)
    client = TestClient(server.app)

    request_data = {
        "model": "minimax-m2-1",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
    }

    # This will likely fail due to upstream connection, but we can test the structure
    response = client.post("/v1/chat/completions", json=request_data)

    # Should either succeed or fail with upstream error
    if response.status_code == 502:
        # Upstream error is expected if vLLM is not running
        assert "Upstream vLLM error" in response.json()["detail"]
    else:
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
