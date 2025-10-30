# Agentic AI without any Agent Frameworks

This is a library for performing LLM-based automation without using complex frameworks.

Using Model Context Protocol - HTTP Streaming for getting more Context 

Just plain MCP and API calls combined with good old-fashioned programming.

A LLM Based Agentic Code review example is given here.

 ## [Code Review MCP Server](code_ast_mcp_server).

See the [Readme](code_ast_mcp_server/README.md) for how a Code Review MCP Server is Implemented

This server contains the business logic for AST parsing and querying the code for function definitions and their call locations.

This setup can be used by an AI agent to generate sufficient context to properly review the code.

## [Code review Agent](llm-mcp-code-review-agent)

See the [Readme](llm-mcp-code-review-agent/README.md) for how a Agentic Code Reivew system can be implemented using the above MCP Server

and plain [programming Idioms](nmagents/command.py) 

The code and containers are also hosted in Hugging Face Spaces ([Code review MCP Server](https://huggingface.co/spaces/alexcpn/code-review-mcp-server/tree/main), [Code Review Agent](https://huggingface.co/spaces/alexcpn/llm-mcp-code-review/tree/main) ) and a Colab Client is provided [here]https://colab.research.google.com/drive/11xryaGH28jpTSd-V2NJ3j5WQJLzr14j4#scrollTo=NRCZqhrb5Pn_

Note - You will need an OpenAI API Key/Subscription for running the Client/Agent or a Local Model say running using OLLAMA

# Running the Full System


## 1. Running the Code AST MCO Server on HTTP

```
cd code_ast_mcp_server/
uv sync 
uv run fastmcp_server.py
```

You can expose the above vi Ngrok `ngrok http http://localhost:7860`

## 2. Running the Code Review Agent

```
cd llm-mcp-code-review-agent
CODE_AST_MCP_SERVER_URL=http://127.0.0.1:7860/mcp uv run uvicorn code_review_agent:app --host 0.0.0.0 --port 8860
```

since we are running the AST MCP Server locally for now


## 3. Running the Test Client

```
cd llm-mcp-code-review-agent
$ uv run  client.py --repo-url https://github.com/huggingface/accelerate --pr-number 3321
```