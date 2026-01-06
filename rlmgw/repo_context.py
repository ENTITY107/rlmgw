"""Repository context collectors for RLMgw."""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RepoContextCollector:
    """Collects context from repository in a read-only, safe manner."""

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).absolute()
        self.excluded_dirs = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}
        self.max_file_size = 1024 * 1024  # 1MB
        self.max_file_read = 1024 * 100  # 100KB per file
        self.max_grep_results = 50

    def _is_excluded(self, path: Path) -> bool:
        """Check if path should be excluded."""
        for part in path.parts:
            if part in self.excluded_dirs:
                return True
        return False

    def _safe_path(self, path: Path) -> Path | None:
        """Ensure path is safe and within repo root."""
        try:
            # Resolve and check if within repo root
            abs_path = path.absolute()
            if not str(abs_path).startswith(str(self.repo_root)):
                logger.warning(f"Path traversal attempt: {path}")
                return None
            if self._is_excluded(abs_path):
                return None
            return abs_path
        except Exception as e:
            logger.warning(f"Invalid path {path}: {e}")
            return None

    def get_repo_fingerprint(self) -> str:
        """Get repository fingerprint using git HEAD or file hashes."""
        try:
            # Try git first
            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if git_head.returncode == 0:
                return git_head.stdout.strip()
        except Exception:
            pass

        # Fallback: hash directory structure
        hasher = hashlib.sha256()
        for root, dirs, files in os.walk(self.repo_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for file in files:
                file_path = Path(root) / file
                if not self._is_excluded(file_path):
                    hasher.update(file.encode())
                    try:
                        with open(file_path, "rb") as f:
                            hasher.update(f.read(self.max_file_size))
                    except Exception:
                        pass

        return hasher.hexdigest()

    def get_repo_tree(self) -> dict[str, Any]:
        """Get repository tree structure."""
        tree = {}

        for root, dirs, files in os.walk(self.repo_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            rel_path = Path(root).relative_to(self.repo_root)
            current_level = tree

            # Navigate to current directory level
            for part in rel_path.parts:
                if not part:
                    continue
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Add files
            for file in files:
                file_path = Path(root) / file
                if not self._is_excluded(file_path):
                    current_level[file] = "file"

        return tree

    def read_file_safe(self, file_path: str) -> str | None:
        """Read file with size limits and path safety checks."""
        # Resolve relative paths against repo_root
        path = self.repo_root / file_path
        path = self._safe_path(path)
        if not path or not path.is_file():
            return None

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read(self.max_file_read)
                return content
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    def grep_repo(
        self, pattern: str, file_extensions: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Search for pattern in repository files."""
        results = {}

        if file_extensions is None:
            file_extensions = [".py", ".md", ".txt", ".json", ".yaml", ".yml"]

        try:
            for root, dirs, files in os.walk(self.repo_root):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

                for file in files:
                    if any(file.endswith(ext) for ext in file_extensions):
                        file_path = Path(root) / file
                        if not self._is_excluded(file_path):
                            content = self.read_file_safe(str(file_path))
                            if content:
                                lines = content.split("\n")
                                matches = [line for line in lines if pattern in line]
                                if matches:
                                    rel_path = str(file_path.relative_to(self.repo_root))
                                    results[rel_path] = matches[: self.max_grep_results]
                                    if len(results) >= self.max_grep_results:
                                        break
        except Exception as e:
            logger.warning(f"Grep failed: {e}")

        return results

    def get_file_list(self, extensions: list[str] | None = None) -> list[str]:
        """Get list of files in repository."""
        file_list = []

        if extensions is None:
            extensions = [".py", ".md", ".txt", ".json", ".yaml", ".yml"]

        for root, dirs, files in os.walk(self.repo_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = Path(root) / file
                    if not self._is_excluded(file_path):
                        rel_path = str(file_path.relative_to(self.repo_root))
                        file_list.append(rel_path)

        return file_list
