"""
# fastmcp_client.py
This is a simple example of how to use the fastmcp client to interact with an MCP server.
It demonstrates how to connect to the server, list available tools, and call specific tools with parameters.

Author: Alex Punnen
License: MIT License

"""
import asyncio
from fastmcp import Client

async def example():
    ## async with Client("http://127.0.0.1:7860/mcp") as client: for local run
    async with Client("https://alexcpn-code-review-mcp-server.hf.space/mcp") as client: for local run
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
        
        repo_url = 'https://github.com/ngrok/ngrok-operator'
        target_name = "createKubernetesOperator"
        # find a specific function
        tool_result = await client.call_tool("get_function_context_for_project_mcp", {"function_name": target_name, "github_repo": repo_url})
        print("Tool result for :get_function_context_for_project_mcp", tool_result)
        tool_result = await client.call_tool("get_function_references_mcp", {"function_name": target_name, "github_repo": repo_url})
        print("Tool result:get_function_references_mcp", tool_result)

if __name__ == "__main__":
    asyncio.run(example())