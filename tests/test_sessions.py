"""Test session management."""

import tempfile

import pytest

from rlmgw.config import RLMgwConfig
from rlmgw.models import SessionData
from rlmgw.sessions import SessionManager


def test_session_management():
    """Test basic session management."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RLMgwConfig()
        config.storage_dir = temp_dir

        session_manager = SessionManager(config)

        # Test session ID generation
        request_data = {"messages": [{"role": "user", "content": "test"}]}
        session_id = session_manager.get_session_id({}, request_data)
        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Test session creation
        session_data = session_manager.create_new_session(session_id, "test_fingerprint")
        assert isinstance(session_data, SessionData)
        assert session_data.session_id == session_id

        # Test session retrieval
        retrieved_session = session_manager.get_session(session_id)
        assert retrieved_session is not None
        assert retrieved_session.session_id == session_id

        # Test session update
        from rlmgw.models import ContextPack

        test_pack = ContextPack(
            repo_fingerprint="test_fingerprint",
            relevant_files=[],
            file_contents={},
            symbols=[],
            constraints=[],
            risks=[],
            suggested_actions=[],
        )
        session_data.context_packs.append(test_pack)
        session_manager.update_session(session_data)

        updated_session = session_manager.get_session(session_id)
        assert len(updated_session.context_packs) == 1


def test_session_id_priority():
    """Test session ID priority (header > request > generated)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = RLMgwConfig()
        config.storage_dir = temp_dir
        session_manager = SessionManager(config)

        request_data = {"messages": [{"role": "user", "content": "test"}]}

        # Test header priority
        headers_with_session = {"x-session-id": "header_session"}
        session_id = session_manager.get_session_id(headers_with_session, request_data)
        assert session_id == "header_session"

        # Test request data priority
        request_data_with_session = {
            "session_id": "request_session",
            "messages": [{"role": "user", "content": "test"}],
        }
        session_id = session_manager.get_session_id({}, request_data_with_session)
        assert session_id == "request_session"

        # Test generated session ID
        session_id = session_manager.get_session_id({}, request_data)
        assert isinstance(session_id, str)
        assert len(session_id) == 16  # SHA256 hash truncated to 16 chars


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
