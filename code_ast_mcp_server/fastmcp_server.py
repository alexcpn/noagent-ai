"""
  FastMCP Server for Code Indexing Tools
    This script sets up a FastMCP server to expose tools for code indexing,
    specifically for retrieving function context, call references, and simple
    string searches within a GitHub repository.
 
    Author: Alex Punnen
    License: MIT License
  
"""
from fastmcp import FastMCP
from tools.code_indexer import (
    get_function_context_for_project,
    find_function_calls_within_project,
    search_codebase_for_project,
)

mcp = FastMCP()


@mcp.tool()
def get_function_context_for_project_mcp(function_name: str, github_repo: str) -> str:
    """
    Get the details of a function in a GitHub repo along with its callees.
    
    @param function_name: The name of the function to find.
    @param github_repo: The URL of the GitHub repo.
    """
    print(f"Finding context for function: {function_name} in repo: {github_repo}")
    return get_function_context_for_project(function_name, github_repo)


@mcp.tool()
def get_function_references_mcp(function_name: str, github_repo: str) -> str:
    """
    Get the references of a function in a GitHub repo.
    
    @param function_name: The name of the function whose references to find.
    @param github_repo: The URL of the GitHub repo.
    """
    print(f"Finding references for function: {function_name} in repo: {github_repo}")
    return find_function_calls_within_project(function_name, github_repo)


@mcp.tool()
def search_codebase_mcp(
    term: str,
    github_repo: str,
    file_patterns: list[str] | None = None,
    ignore_names: list[str] | None = None,
    max_results: int = 200,
) -> str:
    """
    Search the repository for ``term`` and return matching lines in a grep-like format.
    """
    print(f"Searching for term '{term}' in repo: {github_repo}")
    return search_codebase_for_project(
        term=term,
        github_repo=github_repo,
        file_patterns=file_patterns,
        ignore_names=ignore_names,
        max_results=max_results,
    )


if __name__ == "__main__":

    import os
    ip = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    print(f"Starting Code AST MCP server on {ip}:{port}")
    try:
        mcp.run(
            transport="streamable-http",
            host=ip,
            port=port,
            path="/mcp",
            log_level="debug",
        )
    except Exception as e:
        print(f"Failed to start MCP server: {e}")
