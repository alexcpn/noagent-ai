---
title: Code Review MCP Server
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: Dockerfile
pinned: false
---

# MCP Server Exposing Repo Code details

### An MCP Server that provides the context for effective code-review of Pyhton and Go GitHub Repos

MCP is the Model Context Protocol that is used to expose APIs to LLMs, both API description as well as a way to call the API through JSON-RPC.

The MCP server exposes multiple tools implemented in [tools\code_indexer.py](code_ast_mcp_server/tools/code_indexer.py):
- `get_function_context_for_project_mcp` returns the definition and docstring for a function.
- `get_function_references_mcp` lists the call sites for a given function.
- `search_codebase_mcp` performs a lightweight string search across the indexed repository.

The Server use the [TreeSitter project](https://tree-sitter.github.io/tree-sitter/) to do AST parsing of the source and extract, classes, methods, reference and doc stings. Currenly limited to Python files, but can easily extend to other languages that TreeSitter supports

Uses uv as the  package manager.

Client call example in [Colab Notebook](https://colab.research.google.com/drive/11xryaGH28jpTSd-V2NJ3j5WQJLzr14j4#scrollTo=NRCZqhrb5Pn_)

A sample of this server is hosted in Hugging Face Spaces - "<https://alexcpn-code-review-mcp-server.hf.space/mcp/>

This will be used by the llm-mcp-code-review Agent

---

## How to Run

For testing the business logic

```
 uv run code_ast_mcp_server/tools/code_indexer.py 
```

## Running the Server on HTTP

```
cd code_ast_mcp_server/
uv sync 
uv run fastmcp_server.py
```

You can expose the above vi Ngrok `ngrok http http://localhost:7860`

## Building Docker and Running

```
docker build -t codereview-mcp-server .

docker run -it --rm -p 7860:7860 codereview-mcp-server
```
