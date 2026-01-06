"""Configuration management for RLMgw."""

import os
from dataclasses import dataclass


@dataclass
class RLMgwConfig:
    """Configuration for RLMgw server."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8010

    # Upstream vLLM settings
    upstream_base_url: str = "http://localhost:8000/v1"
    upstream_model: str = "minimax-m2-1"
    upstream_connect_timeout: int = 5
    upstream_read_timeout: int = 300
    upstream_max_retries: int = 2

    # Repo settings
    repo_root: str = "."

    # Context pack settings
    max_context_pack_chars: int = 12000
    max_internal_calls: int = 3
    use_rlm_context_selection: bool = True  # Use RLM for intelligent context selection

    # Session settings
    session_ttl_hours: int = 24
    max_sessions: int = 50

    # Storage
    storage_dir: str = ".rlmgw"


def load_config_from_env() -> RLMgwConfig:
    """Load configuration from environment variables."""
    config = RLMgwConfig()

    # Server settings
    if "RLMGW_HOST" in os.environ:
        config.host = os.environ["RLMGW_HOST"]
    if "RLMGW_PORT" in os.environ:
        config.port = int(os.environ["RLMGW_PORT"])

    # Upstream settings
    if "RLMGW_UPSTREAM_BASE_URL" in os.environ:
        config.upstream_base_url = os.environ["RLMGW_UPSTREAM_BASE_URL"]
    if "RLMGW_UPSTREAM_MODEL" in os.environ:
        config.upstream_model = os.environ["RLMGW_UPSTREAM_MODEL"]

    # Repo settings
    if "RLMGW_REPO_ROOT" in os.environ:
        config.repo_root = os.environ["RLMGW_REPO_ROOT"]

    # Context pack settings
    if "RLMGW_MAX_CONTEXT_PACK_CHARS" in os.environ:
        config.max_context_pack_chars = int(os.environ["RLMGW_MAX_CONTEXT_PACK_CHARS"])
    if "RLMGW_MAX_INTERNAL_CALLS" in os.environ:
        config.max_internal_calls = int(os.environ["RLMGW_MAX_INTERNAL_CALLS"])
    if "RLMGW_USE_RLM_CONTEXT_SELECTION" in os.environ:
        config.use_rlm_context_selection = os.environ[
            "RLMGW_USE_RLM_CONTEXT_SELECTION"
        ].lower() in ("true", "1", "yes")

    # Session settings
    if "RLMGW_SESSION_TTL_HOURS" in os.environ:
        config.session_ttl_hours = int(os.environ["RLMGW_SESSION_TTL_HOURS"])
    if "RLMGW_MAX_SESSIONS" in os.environ:
        config.max_sessions = int(os.environ["RLMGW_MAX_SESSIONS"])

    return config


def load_config_from_args(config: RLMgwConfig, args: dict | None = None) -> RLMgwConfig:
    """Load configuration from command line arguments, updating existing config."""
    if args is None:
        return config

    # Override from arguments
    if "host" in args and args["host"]:
        config.host = args["host"]
    if "port" in args and args["port"]:
        config.port = int(args["port"])
    if "repo_root" in args and args["repo_root"]:
        config.repo_root = args["repo_root"]

    return config
