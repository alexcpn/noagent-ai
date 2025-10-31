"""
Author: Alex Punnen
Status:  Demo
This is a simple python based Code Review Agent flow using OpenAI LLM APIs amd Model Context Protocl based client
Design patterns like Command Pattern are used along with for loops to stucture flow and response as we need

"""
import os
import sys
import json
import inspect
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
import requests
import logging as log
from datetime import datetime
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(f"Parent directory: {parentdir}")
# add the parent directory to the system path
sys.path.append(parentdir)
from nmagents.command import CallLLM, ToolCall, ToolList,num_tokens_from_string
import git_uitls

# configure logging

__author__ = "Alex Punnen"
__version__ = "1.0.0"
__email__ = "alexcpn@gmail.com"


#--------------------------------------------------------------------
# Helper functions
#--------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
time_hash = str(datetime.now()).strip()
outfile = "./logs/out_" +  time_hash + "_" + ".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",  #
    # format="[%(levelname)s] %(message)s",  # dont need timing
    handlers=[log.FileHandler(outfile), log.StreamHandler()],
    force=True,
)
# Load the .env file and get the API key
load_dotenv()
#https://platform.openai.com/api-keys add this to your .env file
api_key = os.getenv("OPENAI_API_KEY")
MAX_CONTEXT_LENGTH = 16385
MAX_RETRIES = 5
COST_PER_TOKEN_INPUT =  0.10/10e6 # USD  # https://platform.openai.com/docs/pricing for gpt-4.1-nano
COST_PER_TOKEN_OUTPUT = .40/10e6 # USD
AST_MCP_SERVER_URL = os.getenv(
    "CODE_AST_MCP_SERVER_URL",
    "http://127.0.0.1:7860/mcp",
)

# SEARCH_MCP_SERVER_URL = os.getenv(
#     "CODE_SEARCH_MCP_SERVER_URL",
#     "http://127.0.0.1:7861/mcp",
# )


# Initialize OpenAI client with OpenAI's official base URL
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)
MODEL_NAME= "gpt-4.1-nano"

# openai_client = OpenAI(
#     api_key="sk-local",
#     base_url="http://localhost:11434/v1"
# )
# MODEL_NAME= "phi3.5"


app = FastAPI()


async def main(repo_url,pr_number):

    # Example: get the diff for a specific PR
    print(f"Fetching diffs for PR #{pr_number} from {repo_url}...")
    file_diffs = git_uitls.get_pr_diff_url(repo_url, pr_number)
    print(f"Fetched diffs for {len(file_diffs)} files in PR #{pr_number} from {repo_url}")
    
    #------------------------------------------------
    #  Command to Call the LLM with a budget ( 0.5 Dollars)
    call_llm_command = CallLLM(openai_client, "Call the LLM with the given context", MODEL_NAME, COST_PER_TOKEN_INPUT,COST_PER_TOKEN_OUTPUT, 0.5)
    
    # this this the MCP client invoking the tool - the code review MCP server
    async with Client(AST_MCP_SERVER_URL) as ast_tool_client:
        
        ast_tool_call_command = ToolCall(ast_tool_client, "Call the tool with the given method and params")
        ast_tool_list_command = ToolList(ast_tool_client, "List the available tools")
        # search_tool_call_command = ToolCall(search_tool_client, "Call the tool with the given method and params")
        # search_tool_list_command = ToolList(search_tool_client, "List the available tools") 
        
        ast_tools = await ast_tool_list_command.execute(None)
        #search_tools = await search_tool_list_command.execute(None)

        def _normalize_tools(raw_tools):
            normalized = []
            for item in raw_tools or []:
                if isinstance(item, dict):
                    normalized.append(item)
                elif hasattr(item, "model_dump"):
                    normalized.append(item.model_dump())
                elif hasattr(item, "__dict__"):
                    normalized.append({
                        key: value
                        for key, value in item.__dict__.items()
                        if not key.startswith("_")
                    })
                else:
                    normalized.append(str(item))
            return normalized

        normalized_ast_tools = _normalize_tools(ast_tools)
        #normalized_search_tools = _normalize_tools(search_tools)

        tool_schemas = json.dumps(
            {"ast": normalized_ast_tools,},
            indent=2,
        )
        log.info(f"Available AST tools: {ast_tools}")
        #log.info(f"Available search tools: {search_tools}")
        # Example: log.info diffs for all files (trimmed)
        for file_path, diff in file_diffs.items():
            log.info("-"*80)
            log.info(f"Review code for {file_path}") 
            
            tool_call_example ='{"server": "<ast|lint>", "method": "<method name>", "params": {"<param 1 name>": <param 1 value>, "...": "..."}}'
            main_context =f"""
            You are an expert code reviewer.  You are given the following '{diff}' to review from the repo '{repo_url}' 
            You should generate tool calls to get more context about the code that you are reviewing.
            Whenever you need to look something up— for example, inspect function definitions or call sites—you  you can generate tool calls following the rules below:
            1. **Format**: Every tool call must start with: 'TOOL_CALL:<JSON>'  where `<JSON>` is a valid JSON object matching one of the tool schemas {tool_schemas}
                * Include a field "server" whose value is either "ast" (for the AST tools) or "search" (for the code search tool set).
            2. **No extra text**: Do **not** prepend or append any other words or punctuation to the JSON.
            3. **Once you’ve received the tool result**, continue your reasoning in plain text _without_ re-issuing another TOOL_CALL, unless you need another lookup.
            4. **When you’re done reviewing**, output exactly: DONE: <your final review comments>
            **Example tool call** 
            TOOL_CALL:{tool_call_example}

            """
            
            context = main_context  
            while True:
                response = call_llm_command.execute(context)
                # log.info the response
                log.info(f"LLM response: {response}")
                # Check if the response is a valid JSON
                if response.startswith("TOOL_CALL:"):
                    # Extract the JSON part
                    response = response[len("TOOL_CALL:"):].strip()
                    log.info(f"Extracted JSON: {response}")
                    try:
                        tool_call_payload = json.loads(response)
                    except json.JSONDecodeError as exc:
                        tool_result = f"Invalid JSON response from LLM. Error: {exc}. Original payload: {response}"
                        isSuceess = False
                    else:
                        server_target = str(tool_call_payload.get("server", "ast")).lower()
                        # if server_target == "search":
                        #     command = search_tool_call_command
                        if server_target == "ast":
                            command = ast_tool_call_command
                        else:
                            command = None
                            tool_result = (
                                f"Unknown server '{server_target}'. Please set 'server' to either 'ast' or 'search'."
                            )
                            isSuceess = False
                        if command:
                            tool_result,isSuceess =await command.execute(response)
                    log.info(f"Tool result: {tool_result}")
                    # check before adding to context
                    temp =context + f"Tool call result: {tool_result}"
                    if num_tokens_from_string(temp) < MAX_CONTEXT_LENGTH-10:
                        context = temp
                    else:
                        log.warning("Context too long, not adding tool result to context.")
                elif "DONE" in response:
                    log.info("LLM finished the code review") 
                    log.info("-"*80)
                    break # break out of the loop
                else:
                    # add to the context and continue
                    temp = context + f"LLM response: {response}"
                    if num_tokens_from_string(temp) < MAX_CONTEXT_LENGTH-10:
                        context = temp
                    else:
                        log.info("Context too long, not adding LLM response to context.")
    call_llm_command.get_total_cost()
    return context
    

@app.get("/review")
async def review(repo_url: str, pr_number: int):
    log.info(f"Received review request for {repo_url} PR #{pr_number}")
    try:
        review_comment = await main(repo_url, pr_number)
    except Exception as exc:
        log.exception("Error executing review")
        return JSONResponse(content={"status": "error", "message": str(exc)}, status_code=500)
    return JSONResponse(content={"status": "ok", "review_comment": review_comment or "No review comment produced."})


# 
# if __name__ == "__main__":
#     repo_url = "https://github.com/huggingface/accelerate"
#     pr_number = 2603
#     asyncio.run(main())
