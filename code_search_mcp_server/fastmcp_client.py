"""
Example FastMCP client for the code search server.

Connects to the running MCP endpoint, lists the exposed tools, and performs a
sample search query to demonstrate the request/response flow.
"""

import asyncio

from fastmcp import Client


async def example() -> None:
    """Connect to the server, list tools, and execute a sample search."""
    async with Client("http://127.0.0.1:7861/mcp") as client:
        await client.ping()
        tools = await client.list_tools()
        print("Available tools:", tools)

        parameters = {
            "term": "FastMCP",
            "root": ".",
            "file_patterns": ["*.py"],
            "max_results": 5,
        }
        response = await client.call_tool("search_codebase_mcp", parameters)
        print("Search results:\n", response)


if __name__ == "__main__":
    asyncio.run(example())
