"""Custom REPL environment with repository context tools for RLM."""

import logging
from typing import Any

from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


class RepoContextTools:
    """
    Repository context tools that will be available in the RLM's REPL environment.

    These tools allow the RLM to programmatically explore the codebase.
    """

    def __init__(self, repo_root: str):
        self.collector = RepoContextCollector(repo_root)
        logger.info(f"Initialized RepoContextTools for {repo_root}")

    def list_files(self, extensions: list[str] | None = None) -> list[str]:
        """
        List all files in the repository.

        Args:
            extensions: Optional list of file extensions to filter by (e.g., [".py", ".md"])

        Returns:
            List of file paths relative to repo root
        """
        files = self.collector.get_file_list(extensions)
        logger.debug(f"list_files() returned {len(files)} files")
        return files

    def grep(self, pattern: str, extensions: list[str] | None = None) -> dict[str, list[str]]:
        """
        Search for a pattern across repository files.

        Args:
            pattern: String pattern to search for
            extensions: Optional list of file extensions to search in

        Returns:
            Dictionary mapping file paths to lists of matching lines
        """
        results = self.collector.grep_repo(pattern, extensions)
        logger.debug(f"grep('{pattern}') found matches in {len(results)} files")
        return results

    def read_file(self, path: str) -> str | None:
        """
        Read the contents of a file.

        Args:
            path: Relative path to the file from repo root

        Returns:
            File contents as string, or None if file cannot be read
        """
        content = self.collector.read_file_safe(path)
        if content:
            logger.debug(f"read_file('{path}') returned {len(content)} chars")
        else:
            logger.debug(f"read_file('{path}') failed")
        return content

    def get_tree(self) -> dict[str, Any]:
        """
        Get the directory tree structure of the repository.

        Returns:
            Nested dictionary representing the directory structure
        """
        tree = self.collector.get_repo_tree()
        logger.debug(f"get_tree() returned tree with {len(tree)} top-level items")
        return tree

    def get_fingerprint(self) -> str:
        """
        Get the repository fingerprint (git HEAD or file hash).

        Returns:
            Repository fingerprint string
        """
        fingerprint = self.collector.get_repo_fingerprint()
        logger.debug(f"get_fingerprint() = {fingerprint[:16]}...")
        return fingerprint


def setup_repo_environment_globals(repo_root: str) -> dict[str, Any]:
    """
    Create global namespace for RLM environment with repo context tools.

    This function returns a dictionary that will be injected into the RLM's
    REPL environment as global variables.

    Args:
        repo_root: Path to repository root

    Returns:
        Dictionary of global variables to inject into RLM environment
    """
    repo_tools = RepoContextTools(repo_root)

    # Create convenient globals
    globals_dict = {
        "repo": repo_tools,
        # Also expose individual functions for convenience
        "list_files": repo_tools.list_files,
        "grep": repo_tools.grep,
        "read_file": repo_tools.read_file,
        "get_tree": repo_tools.get_tree,
    }

    logger.info("Created repo environment globals with repo context tools")
    return globals_dict
