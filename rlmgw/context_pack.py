"""Context pack builder for RLMgw."""

import logging

from .models import ContextPack
from .repo_context import RepoContextCollector

logger = logging.getLogger(__name__)


class ContextPackBuilder:
    """Builds structured context packs for RLM recursion."""

    def __init__(self, repo_collector: RepoContextCollector, max_chars: int = 12000):
        self.repo_collector = repo_collector
        self.max_chars = max_chars

    def _truncate_content(self, content: str, max_length: int) -> str:
        """Truncate content to max length."""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "... (truncated)"

    def _build_context_pack(self, query: str, relevant_files: list[str]) -> ContextPack:
        """Build context pack from query and relevant files."""
        repo_fingerprint = self.repo_collector.get_repo_fingerprint()

        # Read relevant files
        file_contents = {}
        total_chars = 0

        for file_path in relevant_files:
            content = self.repo_collector.read_file_safe(file_path)
            if content:
                # Truncate to fit within max_chars limit
                remaining_chars = self.max_chars - total_chars
                if remaining_chars > 0:
                    truncated_content = self._truncate_content(content, remaining_chars)
                    file_contents[file_path] = truncated_content
                    total_chars += len(truncated_content)
                else:
                    break

        # Build context pack
        context_pack = ContextPack(
            repo_fingerprint=repo_fingerprint,
            relevant_files=relevant_files,
            file_contents=file_contents,
            symbols=[],
            constraints=[],
            risks=[],
            suggested_actions=[],
        )

        return context_pack

    def build_from_query(self, query: str) -> ContextPack:
        """Build context pack from user query."""
        # Simple keyword-based file selection
        keywords = self._extract_keywords(query)
        relevant_files = self._find_relevant_files(keywords)

        return self._build_context_pack(query, relevant_files)

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        # Simple keyword extraction - could be enhanced
        words = query.lower().split()
        # Filter out common words
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "and", "or"}
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        return keywords

    def _find_relevant_files(self, keywords: list[str]) -> list[str]:
        """Find files relevant to keywords."""
        if not keywords:
            return []

        # Search for files containing keywords
        relevant_files = set()

        for keyword in keywords:
            grep_results = self.repo_collector.grep_repo(keyword)
            relevant_files.update(grep_results.keys())

        # Also include common project files
        common_files = [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "package.json",
        ]

        for file in common_files:
            if (self.repo_collector.repo_root / file).exists():
                relevant_files.add(file)

        return list(relevant_files)

    def build_from_files(self, file_paths: list[str]) -> ContextPack:
        """Build context pack from specific file paths."""
        return self._build_context_pack("", file_paths)

    def get_context_pack_size(self, context_pack: ContextPack) -> int:
        """Get size of context pack in characters."""
        size = 0
        size += len(context_pack.repo_fingerprint)
        size += sum(len(file) for file in context_pack.relevant_files)
        size += sum(len(content) for content in context_pack.file_contents.values())
        size += sum(len(symbol) for symbol in context_pack.symbols)
        size += sum(len(constraint) for constraint in context_pack.constraints)
        size += sum(len(risk) for risk in context_pack.risks)
        size += sum(len(action) for action in context_pack.suggested_actions)
        return size
