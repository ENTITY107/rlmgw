# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **monorepo containing two related projects**:

1. **`rlm/`** — Recursive Language Models library
   - Core inference engine for RLMs
   - Supports multiple REPL environments (local, docker, modal, prime)
   - Enables LMs to recursively explore and decompose large contexts

2. **`rlmgw/`** — RLM Gateway (OpenAI-compatible)
   - HTTP gateway built on top of `rlm/`
   - Sits in front of vLLM/MiniMax-M2.1
   - Intelligently selects relevant code context using RLM recursion
   - Designed for use with Claude Code to handle extremely large codebases

**Focus**: When working on bugs or features, clarify which component you're targeting.

## Setup & Installation

```bash
# Install dependencies with uv (preferred)
uv sync --group dev --group test

# Or use pip with pyproject.toml
pip install -e ".[dev,test]"

# Install pre-commit hooks (recommended)
uv run pre-commit install
```

**Python Version**: 3.11+ (3.12 recommended)

## Development Commands

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Check and fix linting issues
uv run ruff check --fix .

# Full pre-commit check (before commit)
uv run pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sessions.py -v

# Run single test
uv run pytest tests/test_sessions.py::TestSessionManager -v

# Run with coverage
uv run pytest --cov=rlmgw tests/

# Run tests for core rlm (imports + no circular deps)
uv run pytest tests/test_imports.py -v
```

### Running RLMgw Locally

```bash
# Start RLMgw server (default: localhost:8010)
python3 -m rlmgw.server --host 127.0.0.1 --port 8010 --repo-root .

# Or use the provided script
./scripts/run_rlmgw.sh --host 0.0.0.0 --port 8010 --repo-root /path/to/target-repo

# Test with curl
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m2-1",
    "messages": [{"role": "user", "content": "What is in this repo?"}]
  }'
```

### Environment Variables

```bash
# RLMgw Configuration
export RLMGW_HOST="0.0.0.0"
export RLMGW_PORT="8010"
export RLMGW_REPO_ROOT="/path/to/target-repo"  # Repo to analyze
export RLMGW_UPSTREAM_BASE_URL="http://localhost:8000/v1"  # vLLM endpoint
export RLMGW_UPSTREAM_MODEL="minimax-m2-1"

# Context selection mode
export RLMGW_USE_RLM_CONTEXT_SELECTION="true"  # Use intelligent RLM selection
export RLMGW_MAX_INTERNAL_CALLS="3"  # Max recursive calls for context selection
export RLMGW_MAX_CONTEXT_PACK_CHARS="12000"  # Max context pack size

# Session management
export RLMGW_SESSION_TTL_HOURS="24"
export RLMGW_MAX_SESSIONS="50"
export RLMGW_STORAGE_DIR=".rlmgw"
```

## Architecture

### RLM Library (`rlm/`)

**Core Classes**:
- `RLM` (`core/rlm.py`) — Main entry point; spawns REPL + LM handler per completion
- `BaseEnv` (`environments/base_env.py`) — Base class for REPL environments
- `LocalREPL` (`environments/local_repl.py`) — Default local REPL with safe exec
- `DockerREPL`, `ModalREPL`, `PrimeREPL` — Isolated REPL environments
- `LMHandler` (`core/lm_handler.py`) — Socket-based LM handler for recursive calls

**Flow**:
1. User calls `rlm.completion(prompt)`
2. Spawns REPL environment + LM handler
3. REPL has access to `llm_query()` for recursive sub-calls
4. LM iteratively explores and refines (up to `max_iterations`)
5. REPL returns final answer

**Key Concept**: The LM can execute Python code in the REPL environment, call `llm_query()` for sub-queries, and programmatically explore context.

### RLMgw Gateway (`rlmgw/`)

**Architecture Diagram**:
```
Claude Code
  ↓ (OpenAI-compatible request)
RLMgw Server (FastAPI)
  ↓
Context Selection:
  - RLMContextPackBuilder (if enabled)
    - Uses RLM to explore target repo
    - Repo tools: repo.grep(), repo.read_file(), repo.list_files()
    - Recursive refinement (max_internal_calls)
  - OR ContextPackBuilder (simple keyword matching)
  ↓
Context Pack + User Query
  ↓ (inject into system message)
vLLM Server (OpenAI-compatible)
  ↓ (MiniMax-M2.1)
Response
  ↓
Claude Code
```

**Key Files**:

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server; routes `/v1/chat/completions` |
| `config.py` | Configuration management (env vars + CLI args) |
| `models.py` | Pydantic models (requests/responses) |
| `upstream.py` | HTTP client to vLLM with retries + logging |
| `context_pack_rlm.py` | RLM-based intelligent context selection |
| `context_pack.py` | Simple keyword-based context selection (fallback) |
| `repo_env.py` | Repository exploration tools for RLM REPL |
| `repo_context.py` | Safe read-only access to target repository |
| `sessions.py` | SQLite session management |

**Two Context Selection Modes**:

1. **RLM Mode** (default, `RLMGW_USE_RLM_CONTEXT_SELECTION=true`)
   - Spawns RLM instance to explore target repo
   - RLM uses `repo.*` tools in REPL environment
   - Recursively refines file selection (up to `max_internal_calls`)
   - Best for: Large codebases, complex queries

2. **Simple Mode** (`RLMGW_USE_RLM_CONTEXT_SELECTION=false`)
   - Extracts keywords from query
   - Greps for keywords in target repo
   - Includes common project files
   - Best for: Small codebases, quick responses

## Code Organization & Patterns

### Configuration Loading

Pattern: **Environment variables → CLI args** (both override defaults)

```python
# Load from environment
config = load_config_from_env()
# Override with CLI args (preserves env vars not specified in CLI)
config = load_config_from_args(config, vars(args))
```

### Context Pack Selection

Both selection modes implement the same interface:
- `build_from_query(query: str) -> ContextPack`
- Returns structured context with files, fingerprint, reasoning

Fallback chain:
```
RLM-based (if enabled & available)
  → Simple keyword-based (if RLM fails)
  → Empty context pack (if all fail)
```

### Session Management

Sessions stored in SQLite (`.rlmgw/sessions.db`):
- Identified by: Header (`X-Session-Id`), request field (`session_id`), or hash
- Cached: context packs, repo tree, grep results
- Cleanup: TTL-based + LRU eviction

### Error Handling

Philosophy: **Fail gracefully with detailed logs**

```python
# Good: Detailed error logging
logger.error(f"Failed to select context: {e}")
logger.debug(f"Response body: {response.text}")

# Good: Fallback behavior
if rlm_unavailable:
    logger.warning("RLM not available, using simple selection")
    context_pack = simple_builder.build_from_query(query)

# Good: HTTP errors include response body
except httpx.HTTPStatusError as e:
    logger.error(f"vLLM error:\n{e.response.text}")
```

## Contribution Guidelines

### When Adding Features

1. **Check which component** affects: `rlm/` core or `rlmgw/` gateway?
2. **Follow existing patterns**:
   - Configuration: Use env var + CLI arg pattern
   - Error handling: Log details, fail gracefully
   - Tests: Unit tests in `tests/`, meaningful test names
3. **Run code quality**:
   ```bash
   uv run ruff format . && uv run ruff check --fix . && uv run pytest
   ```
4. **Update relevant docs**:
   - `RLMGW_README.md` for gateway features
   - Docstrings for new classes/methods
   - `CLAUDE.md` if patterns change

### RLM Core (`rlm/`) Guidelines

Per `CONTRIBUTING.md` and project conventions:
- **Avoid touching `core/`** unless necessary — keep the repo minimal
- **Dependencies**: Avoid new core dependencies; use optional extras for non-essential functionality
- **Code style**:
  - No leading `_` for private methods unless explicitly requested
  - Use `cast()`, `assert` for typing; avoid `# type: ignore` without justification
- **Error handling**: Fail fast, fail loud — no silent fallbacks, minimize branching
- **Scope**: Small, surgical diffs — one logical change per PR
- **Dead code**: Delete instead of guarding

### RLMgw Gateway Guidelines

- **Separation of concerns**: Context selection, upstream routing, session management
- **OpenAI compatibility**: Maintain strict compatibility with `/v1/chat/completions` spec
- **Fallback first**: Always have a degraded-but-functional fallback
- **Logging**: Info level for normal flow, warning/error for issues
- **Configuration**: Use environment variables + CLI args pattern
- **RLMgw is additive and isolated**: Do not break or refactor core rlm behavior
- **Non-goals**: No filesystem writes, no shell execution, no Claude Code tool emulation

## Key Files to Know

**For Context Selection Issues**:
- `rlmgw/context_pack_rlm.py` — RLM-based selection logic
- `rlmgw/context_pack.py` — Simple selection fallback
- `rlmgw/repo_env.py` — Repo tools available to RLM

**For Message/Response Issues**:
- `rlmgw/server.py` — Message format conversion (lines 78-106)
- `rlmgw/models.py` — Pydantic request/response models

**For Upstream vLLM Issues**:
- `rlmgw/upstream.py` — HTTP client with retry + logging
- Look for detailed error logs in response body logging

**For Session Issues**:
- `rlmgw/sessions.py` — SQLite session store
- Check `.rlmgw/sessions.db`

## Common Tasks

### Add a New Configuration Option

1. Add field to `RLMgwConfig` dataclass
2. Load from env var in `load_config_from_env()`
3. Load from CLI args in `load_config_from_args()`
4. Use in appropriate component
5. Document in `RLMGW_README.md`

### Debug Context Selection

```bash
# Set log level to DEBUG
export LOGLEVEL=DEBUG

# Run server and inspect logs
python3 -m rlmgw.server --repo-root /path/to/repo

# Test with curl and check context pack selection
curl -X POST http://localhost:8010/v1/chat/completions \
  -d '{"model": "minimax-m2-1", "messages": [{"role": "user", "content": "Find the auth module"}]}'

# Enable/disable RLM mode
export RLMGW_USE_RLM_CONTEXT_SELECTION=false  # Use simple mode
```

### Profile Context Selection Latency

```python
# In context_pack_rlm.py, wrap build_from_query:
import time

def build_from_query(self, query: str) -> ContextPack:
    start = time.perf_counter()
    result = self._build_with_rlm(query)
    elapsed = time.perf_counter() - start
    logger.info(f"Context selection took {elapsed:.2f}s")
    return result
```

### Test with Mock vLLM

RLMgw will fail gracefully if vLLM is unavailable:
- `/readyz` endpoint shows upstream health
- Test with: `curl http://localhost:8010/readyz`
- Consider mocking vLLM for unit tests (see `tests/mock_lm.py`)

## Important Notes

### Recent Improvements (Jan 2026)

Major architecture improvements made:
- Fixed duplicate code in `server.py`
- Fixed config loading to properly merge env vars + CLI args
- Fixed path resolution for repo files
- Added detailed error logging to upstream requests
- **Implemented actual RLM-based context selection** (was previously just keyword matching)
- Added fallback mechanisms and graceful degradation

See `IMPLEMENTATION_REVIEW.md` for complete details.

### Known Limitations

1. **Streaming not supported** — Only non-streaming completions. If streaming is requested, reject with `stream=true not supported` error
2. **RLM mode cost** — Extra LLM calls for context selection (~1-3 per query)
3. **Context truncation** — Large files truncated to fit `max_context_pack_chars`
4. **Single repo** — One RLMgw instance per target repository

### Debugging Tips

- Check logs for which context selection mode is active
- Use `RLMGW_MAX_INTERNAL_CALLS=1` to speed up RLM context selection during testing
- Monitor `.rlmgw/sessions.db` growth (LRU eviction should prevent unbounded growth)
- If vLLM is slow, check its logs for CUDA/memory issues

## References

- **RLM Paper**: https://arxiv.org/abs/2512.24601
- **RLM GitHub**: https://github.com/alexzhang13/rlm
- **RLMgw README**: `RLMGW_README.md`
- **Implementation Review**: `IMPLEMENTATION_REVIEW.md`
- **Architecture Details**: `ARCHITECTURE.md`
