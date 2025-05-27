# Agentic AI without any Agent Frameworks

This is a library for performing LLM-based automation without using complex frameworks.

Using Model Context Protocol - HTTP Streaming for getting more Context 

Just plain MCP and API calls combined with good old-fashioned programming.

A LLM Based Agentic Code review example is given here.

 ## [Code Review MCP Server](code_review_mcp_server).

See the [Readme](code_review_mcp_server/README.md) for how a Code Review MCP Server is Implemented

This server contains the business logic for AST parsing and querying the code for function definitions and their call locations.

This setup can be used by an AI agent to generate sufficient context to properly review the code.

## [Code review Agent](llm-mcp-code-review-agent)

See the [Readme](llm-mcp-code-review-agent/README.md) for how a Agentic Code Reivew system can be implemented using the above MCP Server

and plain [programming Idioms](nmagents/command.py) 

The code and containers are also hosted in Hugging Face Spaces ([Code review MCP Server](https://huggingface.co/spaces/alexcpn/code-review-mcp-server/tree/main), [Code Review Agent](https://huggingface.co/spaces/alexcpn/llm-mcp-code-review/tree/main) ) and a Colab Client is provided [here]https://colab.research.google.com/drive/11xryaGH28jpTSd-V2NJ3j5WQJLzr14j4#scrollTo=NRCZqhrb5Pn_

Note - You will need an OpenAI API Key/Subscription for running the Client/Agent