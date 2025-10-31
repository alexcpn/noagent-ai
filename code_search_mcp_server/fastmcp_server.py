"""
FastMCP server that exposes a source-code search capability backed by simple
file system scanning. The server mirrors the structure of ``code_ast_mcp_server``
so it can be reused alongside the existing tooling.
"""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from tools.file_search import format_matches, search_codebase

mcp = FastMCP()


@mcp.tool()
def search_codebase_mcp(
    term: str,
    root: str = ".",
    file_patterns: list[str] | None = None,
    ignore_names: list[str] | None = None,
    max_results: int = 200,
) -> str:
    """
    Search files under ``root`` for ``term`` and return a grep-like listing.

    Parameters
    ----------
    term:
        The string to search for (case-insensitive).
    root:
        Directory to search. Defaults to the current working directory.
    file_patterns:
        Optional sequence of glob patterns (e.g. ["*.py", "*.ts"]) used to limit
        which files are scanned.
    ignore_names:
        Optional sequence of directory names to ignore in addition to the default set.
    max_results:
        Maximum number of matches returned to the client. Defaults to 200.
    """
    root_path = Path(root).expanduser().resolve()
    matches = search_codebase(
        term=term,
        root=root_path,
        file_patterns=file_patterns,
        ignore_names=ignore_names,
        max_results=max_results,
    )
    return format_matches(matches, root=root_path)


if __name__ == "__main__":
    import os

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7861"))
    print(f"Starting Code Search MCP server on {host}:{port}")
    try:
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            path="/mcp",
            log_level="debug",
        )
    except Exception as exc:
        print(f"Failed to start MCP server: {exc}")
