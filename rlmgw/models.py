"""Pydantic models for RLMgw API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="minimax-m2-1", description="Model to use")
    messages: list[dict[str, str]] = Field(description="List of messages")
    temperature: float | None = Field(default=0.7, description="Temperature for sampling")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    stream: bool | None = Field(default=False, description="Whether to stream response")
    session_id: str | None = Field(default=None, description="Session ID for context continuity")


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""

    index: int = Field(description="Index of the choice")
    message: dict[str, Any] = Field(description="Message content")
    finish_reason: str = Field(description="Reason for finishing")


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(description="Number of tokens in prompt")
    completion_tokens: int = Field(description="Number of tokens in completion")
    total_tokens: int = Field(description="Total tokens used")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(description="Completion ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    choices: list[ChatCompletionChoice] = Field(description="List of choices")
    usage: UsageInfo | None = Field(default=None, description="Token usage information")


class ContextPack(BaseModel):
    """Structured context pack for RLM recursion."""

    repo_fingerprint: str = Field(description="Repository fingerprint")
    relevant_files: list[str] = Field(default_factory=list, description="Relevant file paths")
    file_contents: dict[str, str] = Field(default_factory=dict, description="File contents")
    symbols: list[str] = Field(default_factory=list, description="Relevant symbols/APIs")
    constraints: list[str] = Field(default_factory=list, description="Constraints/invariants")
    risks: list[str] = Field(default_factory=list, description="Risks/edge cases")
    suggested_actions: list[str] = Field(default_factory=list, description="Suggested next actions")


class SessionData(BaseModel):
    """Session data stored in database."""

    session_id: str = Field(description="Session ID")
    created_at: datetime = Field(description="Creation timestamp")
    last_used: datetime = Field(description="Last used timestamp")
    repo_fingerprint: str = Field(description="Repository fingerprint")
    context_packs: list[ContextPack] = Field(
        default_factory=list, description="Cached context packs"
    )
    repo_tree: dict[str, Any] | None = Field(default=None, description="Cached repo tree")
    grep_cache: dict[str, Any] | None = Field(default=None, description="Cached grep results")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Status")
    timestamp: datetime = Field(description="Timestamp")
    version: str = Field(description="RLMgw version")


class ReadyResponse(BaseModel):
    """Readiness check response."""

    status: str = Field(description="Status")
    timestamp: datetime = Field(description="Timestamp")
    upstream_healthy: bool = Field(description="Upstream vLLM health")
    upstream_model: str = Field(description="Upstream model")
