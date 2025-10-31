"""Utility functions for searching code bases using plain file system scanning.

The search logic intentionally avoids shelling out to external tools so it works
wherever the MCP server is deployed. Results are returned in a grep-like format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pyc",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
)


@dataclass
class Match:
    """A single match record returned to the MCP client."""

    file_path: Path
    line_number: int
    content: str

    def format(self, root: Path) -> str:
        try:
            rel_path = self.file_path.relative_to(root)
        except ValueError:
            rel_path = self.file_path
        return f"{rel_path}:{self.line_number}: {self.content}"


def _iter_files(
    root: Path,
    file_globs: Sequence[str] | None,
    ignores: Sequence[str],
) -> Iterable[Path]:
    """Yield candidate files under ``root`` honouring inclusion and exclusion rules."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        parent_parts = set(path.parts[:-1])
        if any(name in parent_parts for name in ignores):
            continue

        if file_globs:
            if not any(path.match(pattern) for pattern in file_globs):
                continue

        yield path


def _is_binary(content: bytes) -> bool:
    """Heuristic to skip binary files."""
    text_sample = content[:1024]
    if b"\x00" in text_sample:
        return True
    return False


def search_codebase(
    term: str,
    root: str | Path,
    *,
    file_patterns: Sequence[str] | None = None,
    ignore_names: Sequence[str] | None = None,
    max_results: int = 200,
) -> list[Match]:
    """Search ``root`` directory for ``term`` and return up to ``max_results`` matches."""
    if not term:
        raise ValueError("Search term must not be empty")

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Search root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Search root must be a directory: {root_path}")

    ignores = list(DEFAULT_IGNORES)
    if ignore_names:
        ignores.extend(ignore_names)

    normalized_term = term.lower()
    matches: list[Match] = []

    for file_path in _iter_files(root_path, file_patterns, ignores):
        try:
            content_bytes = file_path.read_bytes()
        except (OSError, PermissionError):
            continue

        if _is_binary(content_bytes):
            continue

        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content_bytes.decode("latin-1")
            except UnicodeDecodeError:
                continue

        for idx, line in enumerate(text.splitlines(), start=1):
            if normalized_term in line.lower():
                matches.append(Match(file_path=file_path, line_number=idx, content=line))
                if len(matches) >= max_results:
                    return matches

    return matches


def format_matches(matches: Sequence[Match], root: Path) -> str:
    """Convert a sequence of matches to a printable string for MCP responses."""
    if not matches:
        return "No matches found."

    formatted = "\n".join(match.format(root) for match in matches)
    return formatted
