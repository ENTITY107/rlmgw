"""FastAPI server for RLMgw."""

import argparse
import logging
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request

from .config import RLMgwConfig, load_config_from_args, load_config_from_env
from .context_pack import ContextPackBuilder
from .context_pack_rlm import RLMContextPackBuilder
from .models import (
    ChatCompletionRequest,
    ContextPack,
    HealthResponse,
    ReadyResponse,
)
from .repo_context import RepoContextCollector
from .sessions import SessionManager
from .upstream import UpstreamClient

logger = logging.getLogger(__name__)

# Try to import RLM context pack builder (may fail if RLM not available)
try:
    from .context_pack_rlm import RLMContextPackBuilder

    RLM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RLM context pack builder not available: {e}")
    RLM_AVAILABLE = False


class RLMgwServer:
    """Main RLMgw server class."""

    def __init__(self, config: RLMgwConfig):
        self.config = config
        self.app = FastAPI(title="RLMgw", version="0.1.0")

        # Initialize components
        self.upstream_client = UpstreamClient(config)
        self.repo_collector = RepoContextCollector(config.repo_root)

        # Choose context pack builder based on configuration
        if config.use_rlm_context_selection and RLM_AVAILABLE:
            logger.info("Using RLM-based intelligent context selection")
            self.context_pack_builder = RLMContextPackBuilder(self.repo_collector, config)
        else:
            if config.use_rlm_context_selection and not RLM_AVAILABLE:
                logger.warning(
                    "RLM context selection requested but not available. Using simple selection."
                )
            else:
                logger.info("Using simple keyword-based context selection")
            self.context_pack_builder = ContextPackBuilder(
                self.repo_collector, config.max_context_pack_chars
            )

        self.session_manager = SessionManager(config)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/healthz", response_model=HealthResponse)
        async def healthz():
            """Health check endpoint."""
            return HealthResponse(status="healthy", timestamp=datetime.now(), version="0.1.0")

        @self.app.get("/readyz", response_model=ReadyResponse)
        async def readyz():
            """Readiness check endpoint."""
            upstream_healthy = self.upstream_client.health_check()
            return ReadyResponse(
                status="ready" if upstream_healthy else "not_ready",
                timestamp=datetime.now(),
                upstream_healthy=upstream_healthy,
                upstream_model=self.config.upstream_model,
            )

        @self.app.post("/v1/chat/completions", include_in_schema=False)
        async def chat_completions_raw(request: Request, x_session_id: str = Header(None)):
            """Raw chat completions endpoint that handles both OpenAI and Claude Code formats."""

            logger.info("Received request to /v1/chat/completions")

            # Get raw request body
            try:
                raw_data = await request.json()
                logger.info(f"Raw request data: {raw_data}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}") from e

            # Handle message format conversion for Claude Code
            if "messages" in raw_data and isinstance(raw_data["messages"], list):
                messages = raw_data["messages"]

                # Check if any message has complex content structure
                conversion_done = False
                for i, msg in enumerate(messages):
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
                        if isinstance(content, list):
                            # Convert complex content to simple string
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    else:
                                        # Log non-text content types that are being dropped
                                        logger.warning(
                                            f"Dropping non-text content type '{item.get('type')}' "
                                            f"in message {i}. Content: {str(item)[:100]}"
                                        )
                            messages[i]["content"] = " ".join(text_parts)
                            conversion_done = True

                # Update the raw data with converted messages
                if conversion_done:
                    raw_data["messages"] = messages
                    logger.info("Converted Claude Code message format")

            # Now validate with Pydantic
            try:
                chat_request = ChatCompletionRequest(**raw_data)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid request format: {str(e)}"
                ) from e

            # Rest of the original logic...
            # Reject streaming requests
            if chat_request.stream:
                raise HTTPException(status_code=400, detail="Streaming is not supported")

            # Get session ID
            request_headers = dict(request.headers)
            request_data = chat_request.model_dump()
            session_id = self.session_manager.get_session_id(request_headers, request_data)

            # Get or create session
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                repo_fingerprint = self.repo_collector.get_repo_fingerprint()
                session_data = self.session_manager.create_new_session(session_id, repo_fingerprint)

            # Build context pack from query
            user_query = chat_request.messages[-1]["content"] if chat_request.messages else ""
            context_pack = self.context_pack_builder.build_from_query(user_query)

            # Update session with new context pack
            session_data.context_packs.append(context_pack)
            session_data.last_used = datetime.now()
            self.session_manager.update_session(session_data)

            # Prepare request for upstream vLLM
            # Include context pack in system message
            system_message = {"role": "system", "content": self._format_context_pack(context_pack)}

            messages = [system_message] + chat_request.messages
            upstream_request = ChatCompletionRequest(
                model=chat_request.model,
                messages=messages,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                stream=False,  # Never stream to upstream
            )

            # Get completion from upstream
            try:
                response = self.upstream_client.chat_completion(upstream_request)
                return response
            except Exception as e:
                logger.error(f"Upstream request failed: {e}")
                raise HTTPException(status_code=502, detail=f"Upstream vLLM error: {str(e)}") from e

    def _format_context_pack(self, context_pack: ContextPack) -> str:
        """Format context pack for inclusion in system message."""
        formatted = "### Repository Context\n\n"
        formatted += f"Repository Fingerprint: {context_pack.repo_fingerprint}\n\n"

        if context_pack.relevant_files:
            formatted += "Relevant Files:\n"
            for file in context_pack.relevant_files:
                formatted += f"- {file}\n"
            formatted += "\n"

        if context_pack.file_contents:
            formatted += "File Contents:\n"
            for file, content in context_pack.file_contents.items():
                formatted += f"\n--- {file} ---\n"
                formatted += content
                formatted += "\n---\n\n"

        return formatted

    def run(self, host: str, port: int):
        """Run the server."""
        uvicorn.run(self.app, host=host, port=port, log_level="info")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RLMgw - RLM Gateway Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to listen on")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")

    args = parser.parse_args()

    # Load configuration (env vars first, then CLI args override)
    config = load_config_from_env()
    config = load_config_from_args(config, vars(args))

    # Initialize and run server
    server = RLMgwServer(config)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
