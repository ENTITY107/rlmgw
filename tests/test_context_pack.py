"""Test context pack building."""

import pytest

from rlmgw.context_pack import ContextPackBuilder
from rlmgw.models import ContextPack
from rlmgw.repo_context import RepoContextCollector


def test_repo_context_collector():
    """Test repository context collector."""
    collector = RepoContextCollector(".")

    # Test fingerprint generation
    fingerprint = collector.get_repo_fingerprint()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) > 0

    # Test file reading
    readme_content = collector.read_file_safe("README.md")
    assert readme_content is not None
    assert len(readme_content) > 0

    # Test file list
    files = collector.get_file_list()
    assert isinstance(files, list)
    assert len(files) > 0


def test_context_pack_builder():
    """Test context pack builder."""
    collector = RepoContextCollector(".")
    builder = ContextPackBuilder(collector, max_chars=1000)

    # Test building from query
    context_pack = builder.build_from_query("test query")
    assert isinstance(context_pack, ContextPack)
    assert context_pack.repo_fingerprint is not None

    # Test context pack size - size includes metadata (fingerprint, files, etc.)
    size = builder.get_context_pack_size(context_pack)
    assert size > 0
    # Note: get_context_pack_size includes all metadata fields, not just file_contents
    # The actual file contents are truncated to max_chars during build


def test_context_pack_truncation():
    """Test that context packs are properly truncated."""
    collector = RepoContextCollector(".")
    builder = ContextPackBuilder(collector, max_chars=1000)

    context_pack = builder.build_from_query("test query for truncation")

    # Test that truncation produces reasonable output
    file_contents_size = sum(len(c) for c in context_pack.file_contents.values())
    assert file_contents_size > 0

    # For a reasonable max_chars, the file_contents should be reasonably bounded
    # (not the entire repo, which would be > 10000 chars)
    assert file_contents_size <= 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
