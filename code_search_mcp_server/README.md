# Code Search MCP Server

This directory contains a lightweight [FastMCP](https://pypi.org/project/fastmcp/) server that exposes a single tool for searching a source tree. It mirrors the structure of `code_ast_mcp_server` so it can be deployed alongside the existing code-review tooling.

## Setup

```bash
cd code_search_mcp_server
uv sync
```

## Run

```bash
uv run fastmcp_server.py
```

By default the server listens on `127.0.0.1:7861` with the MCP path `/mcp`. You can override the host or port via the `HOST` and `PORT` environment variables.

## Exposed Tool

`search_codebase_mcp(term, root=".", file_patterns=None, ignore_names=None, max_results=200)`  
Search for a string (case-insensitive) under the supplied root directory and return results similar to `grep`. You can constrain the search with glob patterns and pass additional directory names to ignore. Matches are capped at `max_results` to keep responses manageable.

