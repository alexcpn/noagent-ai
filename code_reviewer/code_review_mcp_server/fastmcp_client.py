import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:4200/mcp") as client:
        await client.ping()
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", tools)
        repo_url = "https://github.com/kurtmckee/feedparser"
        # find a specific function
        target_name = "_start_dc_contributor"
        tool_result = await client.call_tool("get_function_context_for_project_mcp", {"function_name": target_name, "github_repo": repo_url})
        print("Tool result for :get_function_context_for_project_mcp", tool_result)
        tool_result = await client.call_tool("get_function_references_mcp", {"function_name": target_name, "github_repo": repo_url})
        print("Tool result:get_function_references_mcp", tool_result)

if __name__ == "__main__":
    asyncio.run(example())