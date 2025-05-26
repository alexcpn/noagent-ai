---
title: Code Review MCP Server
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: Dockerfile
pinned: false
---

This is hosted in HuggingFace Spaces for now
https://huggingface.co/spaces/alexcpn/code_review_mcp_server/tree/main

And the Server is running at https://alexcpn-code-review-mcp-server.hf.space/mcp

You can Run the MCP Client like below:

[MCP Client Colab Notebook](https://colab.research.google.com/drive/11xryaGH28jpTSd-V2NJ3j5WQJLzr14j4?usp=sharing)

```
 async with Client("https://alexcpn-code-review-mcp-server.hf.space/mcp") as client:
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
        
```