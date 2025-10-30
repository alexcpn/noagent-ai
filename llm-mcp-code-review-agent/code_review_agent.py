"""
Author: Alex Punnen
Status:  Demo
This is a simple python based Code Review Agent flow using OpenAI LLM APIs amd Model Context Protocl based client
Design patterns like Command Pattern are used along with for loops to stucture flow and response as we need

"""
import os
import sys
import inspect
import asyncio
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
import requests
import re
from collections import defaultdict
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
MCP_SERVER_URL = os.getenv(
    "CODE_REVIEW_MCP_SERVER_URL",
    "http://127.0.0.1:7860/mcp",
)

# Initialize OpenAI client with OpenAI's official base URL
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)
app = FastAPI()
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")  # GitLab personal access token

def get_pr_diff_url(repo_url, pr_number):
    """ 
    Get the diff URL for a specific pull request number.
    Args:
        repo_url (str): The URL of the GitHub repository.
        pr_number (int): The pull request number.
    """
    pr_diff_url = f"https://patch-diff.githubusercontent.com/raw/{repo_url.split('/')[-2]}/{repo_url.split('/')[-1]}/pull/{pr_number}.diff"
    response = requests.get(pr_diff_url)

    if response.status_code != 200:
        log.info(f"Failed to fetch diff: {response.status_code}")
        exit()

    if response.status_code != 200:
        log.info(f"Failed to fetch diff: {response.status_code}")
        exit()

    diff_text = response.text
    file_diffs = defaultdict(str)
    file_diff_pattern = re.compile(r'^diff --git a/(.*?) b/\1$', re.MULTILINE)
    split_points = list(file_diff_pattern.finditer(diff_text))
    for i, match in enumerate(split_points):
        file_path = match.group(1)
        start = match.start()
        end = split_points[i + 1].start() if i + 1 < len(split_points) else len(diff_text)
        file_diffs[file_path] = diff_text[start:end]
    return file_diffs
    

async def main(repo_url,pr_number):

    # Example: get the diff for a specific PR
    print(f"Fetching diffs for PR #{pr_number} from {repo_url}...")
    file_diffs = get_pr_diff_url(repo_url, pr_number)
    print(f"Fetched diffs for {len(file_diffs)} files in PR #{pr_number} from {repo_url}")
    
    #------------------------------------------------
    #  Command to Call the LLM with a budget ( 0.5 Dollars)
    call_llm_command = CallLLM(openai_client, "Call the LLM with the given context", "gpt-4.1-nano", COST_PER_TOKEN_INPUT,COST_PER_TOKEN_OUTPUT, 0.5)
    
    # this this the MCP client invoking the tool - the code review MCP server
    async with Client(MCP_SERVER_URL) as fastmcp_client:
        tool_call_command = ToolCall(fastmcp_client, "Call the tool with the given method and params")
        tool_list_command = ToolList(fastmcp_client, "List the available tools")
        
        tools = await tool_list_command.execute(None)
        log.info(f"Available tools: {tools}")
        # Example: log.info diffs for all files (trimmed)
        for file_path, diff in file_diffs.items():
            log.info("-"*80)
            log.info(f"Review diff for {file_path}") 
            
            # main_context = f"You are an expert Python code reviewer, You are given the following {diff} to review from the repo {repo_url} " + \
            # f"You can use the following tools {tools} if needed to get more context about the code that you are reviewing," + \
            # "if you need to check the functions used in the code, or where they are called  you can call the tools" + \
            # f"For framing a call to the tool you can use the format of the tool '{tools}'. Frame the JSON RPC call to the tool" +  \
            # "If you need to call the tool start response with TOOL_CALL:<json format for the tool call>" + \
            # "here is the JSON RPC call format {{\"method\": \"<method name>\", \"params\": {{\"<param 1 name>\": {<param 1 value>}, \"<param 2 name>\": {<param 2 value>} etc }}}}" +\
            # "If you have finished with the review you can start your response with 'DONE:' and give the final review comments "
            tool_call_example ='{{"method\": \"<method name>\", \"params\": {{\"<param 1 name>\": {<param 1 value>}, \"<param 2 name>\": {<param 2 value>} etc }}}}'
            main_context =f"""
            You are an expert code reviewer.  You are given the following '{diff}' to review from the repo '{repo_url}' 
            You should generate tool calls to get more context about the code that you are reviewing.
            Whenever you need to look something up— for example, inspect function definitions or call sites—you  you can generate tool calls following the rules below:
            1. **Format**: Every tool call must start with: 'TOOL_CALL:<JSON>'  where `<JSON>` is a valid JSON object matching one of the tool schemas {tools}
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
                    tool_result,isSuceess =await tool_call_command.execute(response)
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


@app.route("/webhook", methods=["POST"])
async def webhook(request: Request, x_github_event: str = Header(...)):
    try:
        x_github_event = request.headers.get("X-GitHub-Event")
        log.info(f"Received webhook event: {x_github_event}")
        data = await request.json()
    except Exception as e:
        log.error(f"Error parsing JSON: {e}")
        return JSONResponse(content={"status": "error", "message": "Invalid JSON"}, status_code=400)
    log.info(f"Webhook data: {data}")
        # Handle PR review comment events
    if x_github_event == "pull_request_review_comment":
        comment_body = data.get("comment", {}).get("body", "")
        if "@code_review" in comment_body:
            repo_full_name = data["repository"]["full_name"]               # e.g. alexcpn/accelerate-test
            pr_url = data["comment"]["pull_request_url"]                   # e.g. .../pulls/1
            pr_number = int(pr_url.split("/")[-1])
            repo_url = f"https://github.com/{repo_full_name}"

            log.info(f"Triggered code review on {repo_url} PR #{pr_number}")

            review_comment = await main(repo_url, pr_number) or "No issues found."

            # Post back to the same thread
            comment_url = data["comment"]["url"]
            headers = {
                "Authorization": f"token {GITLAB_TOKEN}",
                "Accept": "application/vnd.github+json"
            }
            post_response = requests.post(
                comment_url,
                headers=headers,
                json={"body": f"AI Code Review:\n```\n{review_comment}\n```"}
            )
            log.info(f"Posted review result: {post_response.status_code}")
            return JSONResponse(content={"status": "review triggered"})
        
    return JSONResponse(content={"status": "ok"})
# 
# if __name__ == "__main__":
#     repo_url = "https://github.com/huggingface/accelerate"
#     pr_number = 2603
#     asyncio.run(main())
