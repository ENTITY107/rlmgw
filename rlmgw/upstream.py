"""Upstream vLLM client for RLMgw."""

import logging
from typing import Any

import httpx

from .config import RLMgwConfig
from .models import ChatCompletionRequest, ChatCompletionResponse

logger = logging.getLogger(__name__)


class UpstreamClient:
    """Client for communicating with upstream vLLM server."""

    def __init__(self, config: RLMgwConfig):
        self.config = config
        self.client = httpx.Client(
            base_url=config.upstream_base_url,
            timeout=httpx.Timeout(
                connect=config.upstream_connect_timeout,
                read=config.upstream_read_timeout,
                write=None,
                pool=None,
            ),
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _make_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Make a request to upstream vLLM with retry logic."""
        for attempt in range(self.config.upstream_max_retries):
            try:
                response = self.client.post(
                    "/chat/completions",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # Log response body for HTTP errors
                error_body = e.response.text if hasattr(e, "response") else "No response body"
                if attempt == self.config.upstream_max_retries - 1:
                    logger.error(
                        f"Upstream request failed after {self.config.upstream_max_retries} attempts: {e}\n"
                        f"Response body: {error_body}"
                    )
                    raise
                logger.warning(
                    f"Upstream request attempt {attempt + 1} failed, retrying: {e}\n"
                    f"Response body: {error_body}"
                )
            except httpx.HTTPError as e:
                # Other HTTP errors (timeout, connection, etc.)
                if attempt == self.config.upstream_max_retries - 1:
                    logger.error(
                        f"Upstream request failed after {self.config.upstream_max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(f"Upstream request attempt {attempt + 1} failed, retrying: {e}")

        raise Exception("Upstream request failed after maximum retries")

    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Get chat completion from upstream vLLM."""
        # Ensure we use the configured model
        request_data = request.model_dump()
        request_data["model"] = self.config.upstream_model

        # Don't support streaming to upstream
        request_data["stream"] = False

        response_data = self._make_request(request_data)

        # Convert to our response model
        return ChatCompletionResponse(**response_data)

    def health_check(self) -> bool:
        """Check if upstream vLLM is healthy."""
        try:
            response = self.client.get("/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
