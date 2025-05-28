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

The MCP server exposes a method that gives the context of a Python function  and details about it including callee.
This is in [tools\code_indexer.py](code_review_mcp_server/tools/code_indexer.py)

This tool is available to the LLM to get context for reviewing code snippet given to it in the repo

The Server use the [TreeSitter project](https://tree-sitter.github.io/tree-sitter/) to do AST parsing of the source and extract, classes, methods, reference and doc stings. Currenly limited to Python files, but can easily extend to other languages that TreeSitter supports

Uses uv as the  package manager.

Client call example in [Colab Notebook](https://colab.research.google.com/drive/11xryaGH28jpTSd-V2NJ3j5WQJLzr14j4#scrollTo=NRCZqhrb5Pn_)

A sample of this server is hosted in Hugging Face Spaces - "<https://alexcpn-code-review-mcp-server.hf.space/mcp/>

This will be used by the llm-mcp-code-review Agent

---

## How to Run

For testing the business logic

```
 uv run code_review_mcp_server/tools/code_indexer.py 
```

## Running the Server on HTTP

```
cd code_review_mcp_server/
uv sync 
uv run fastmcp_server.py
```

You can expose the above vi Ngrok `ngrok http http://localhost:7860`

## Building Docker and Running

```
docker build -t codereview-mcp-server .

docker run -it --rm -p 7860:7860 codereview-mcp-server
```
