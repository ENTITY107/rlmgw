"""Session management for RLMgw."""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import RLMgwConfig
from .models import ContextPack, SessionData

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions with SQLite persistence."""

    def __init__(self, config: RLMgwConfig):
        self.config = config
        self.storage_dir = Path(config.storage_dir)
        self.db_path = self.storage_dir / "sessions.db"

        # Ensure storage directory exists
        self.storage_dir.mkdir(exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    repo_fingerprint TEXT NOT NULL,
                    context_packs TEXT NOT NULL,
                    repo_tree TEXT,
                    grep_cache TEXT
                )
            """)

            # Create index for faster lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON sessions(last_used)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_repo_fingerprint ON sessions(repo_fingerprint)"
            )

            conn.commit()

    def _cleanup_old_sessions(self):
        """Clean up old sessions based on TTL."""
        ttl = timedelta(hours=self.config.session_ttl_hours)
        cutoff = datetime.now() - ttl

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM sessions
                WHERE datetime(last_used) < datetime(?)
            """,
                (cutoff.isoformat(),),
            )

            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old sessions")

            conn.commit()

    def _enforce_max_sessions(self):
        """Enforce maximum number of sessions with LRU eviction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count current sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]

            if count > self.config.max_sessions:
                # Delete oldest sessions
                to_delete = count - self.config.max_sessions
                cursor.execute(
                    """
                    DELETE FROM sessions
                    WHERE session_id IN (
                        SELECT session_id FROM sessions
                        ORDER BY last_used ASC
                        LIMIT ?
                    )
                """,
                    (to_delete,),
                )

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Evicted {deleted_count} sessions to enforce max limit")

                conn.commit()

    def _generate_session_id(self, request_data: dict[str, Any]) -> str:
        """Generate deterministic session ID from request data."""
        # Create hash from request data
        data_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_session_id(self, request_headers: dict[str, str], request_data: dict[str, Any]) -> str:
        """Get session ID from headers, request data, or generate new one."""
        # Prefer X-Session-Id header
        if "x-session-id" in request_headers:
            return request_headers["x-session-id"]

        # Fallback to session_id in request data
        if "session_id" in request_data and request_data["session_id"]:
            return request_data["session_id"]

        # Generate deterministic session ID
        return self._generate_session_id(request_data)

    def get_session(self, session_id: str) -> SessionData | None:
        """Get session data by ID."""
        self._cleanup_old_sessions()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, created_at, last_used, repo_fingerprint,
                       context_packs, repo_tree, grep_cache
                FROM sessions
                WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            (
                session_id,
                created_at,
                last_used,
                repo_fingerprint,
                context_packs_json,
                repo_tree_json,
                grep_cache_json,
            ) = row

            # Parse JSON data
            context_packs = [ContextPack(**pack) for pack in json.loads(context_packs_json)]
            repo_tree = json.loads(repo_tree_json) if repo_tree_json else None
            grep_cache = json.loads(grep_cache_json) if grep_cache_json else None

            return SessionData(
                session_id=session_id,
                created_at=datetime.fromisoformat(created_at),
                last_used=datetime.fromisoformat(last_used),
                repo_fingerprint=repo_fingerprint,
                context_packs=context_packs,
                repo_tree=repo_tree,
                grep_cache=grep_cache,
            )

    def update_session(self, session_data: SessionData):
        """Update session data."""
        self._cleanup_old_sessions()
        self._enforce_max_sessions()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Serialize data to JSON
            context_packs_json = json.dumps(
                [pack.model_dump() for pack in session_data.context_packs]
            )
            repo_tree_json = json.dumps(session_data.repo_tree) if session_data.repo_tree else None
            grep_cache_json = (
                json.dumps(session_data.grep_cache) if session_data.grep_cache else None
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, created_at, last_used, repo_fingerprint,
                    context_packs, repo_tree, grep_cache
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_data.session_id,
                    session_data.created_at.isoformat(),
                    session_data.last_used.isoformat(),
                    session_data.repo_fingerprint,
                    context_packs_json,
                    repo_tree_json,
                    grep_cache_json,
                ),
            )

            conn.commit()

    def create_new_session(self, session_id: str, repo_fingerprint: str) -> SessionData:
        """Create new session."""
        now = datetime.now()

        session_data = SessionData(
            session_id=session_id,
            created_at=now,
            last_used=now,
            repo_fingerprint=repo_fingerprint,
            context_packs=[],
            repo_tree=None,
            grep_cache=None,
        )

        self.update_session(session_data)
        return session_data

    def get_session_count(self) -> int:
        """Get current number of active sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]
