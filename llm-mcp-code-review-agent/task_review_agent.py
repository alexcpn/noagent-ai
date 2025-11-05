"""
Author: Alex Punnen
Status:  Demo
This is a simple python based Code Review Agent flow using OpenAI LLM APIs amd Model Context Protocl based client
Design patterns like Command Pattern are used along with for loops to stucture flow and response as we need

"""
import git_utils
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
import logging as log
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import sys
import json
import inspect
import yaml
import re

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
AST_MCP_SERVER_URL = os.getenv(
    "CODE_AST_MCP_SERVER_URL",
    "http://127.0.0.1:7860/mcp",
)

SEARCH_MCP_SERVER_URL = os.getenv(
    "CODE_SEARCH_MCP_SERVER_URL",
    "http://127.0.0.1:7861/mcp",
)


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

def extract_code_blocks(text: str) -> list[str]:
    # Match ```lang (optional) ... ``` with DOTALL so newlines are included.
    pattern = r"```(?:\w+\n)?(.*?)```"
    return [block.strip() for block in re.findall(pattern, text, re.S)]

async def main(repo_url,pr_number):

    # Example: get the diff for a specific PR
    print(f"Code review for PR #{pr_number} from {repo_url}...")

    
    #------------------------------------------------
    #  Command to Call the LLM with a budget ( 0.5 Dollars)
    call_llm_command = CallLLM(openai_client, "Call the LLM with the given context", MODEL_NAME, COST_PER_TOKEN_INPUT,COST_PER_TOKEN_OUTPUT, 0.5)
    
    # this this the MCP client invoking the tool - the code review MCP server
    async with Client(AST_MCP_SERVER_URL) as ast_tool_client:
        
        # ast_tool_call_command = ToolCall(ast_tool_client, "Call the tool with the given method and params")
        # ast_tool_list_command = ToolList(ast_tool_client, "List the available tools")
       
        # read the task schema.yaml file
        sample_task_schema_file = "task_schema.yaml"
        with open(sample_task_schema_file, "r", encoding="utf-8") as f:
            task_schema_content = f.read()
        
        sample_step_schema_file = "steps_schema.yaml"
        log.info(f"Using step schema file: {sample_step_schema_file}")
        with open(sample_step_schema_file, "r", encoding="utf-8") as f:
            step_schema_content = f.read()
      
        tool_schemas = "tools.yaml"
        log.info(f"Using  tools file: {tool_schemas}")
        with open(tool_schemas, "r", encoding="utf-8") as f:
            tool_schemas_content = f.read()
                  
        file_diffs = git_utils.get_pr_diff_url(repo_url, pr_number)
       
        main_context =f"""
        Your task today is Code Reivew. You are given the following '{pr_number}' to review from the repo '{repo_url}' 
        You have to first come up with a plan to review the code changes in the PR as a series of steps.
        For the review, check for adherence to code standards, correctness of implementation, test coverage,
        and impact on existing functionality.   You can split these also as different steps in your plan.
        Write the plan as per the following step schema: {step_schema_content}
        Make sure to follow the step schema format exactly and output only the yaml between codeblocks ``` and  ```.
        """
        log.info("-"*80)
        log.info(f"Generating code review plan for PR #{pr_number} from {repo_url}...{main_context}")
        log.info("-"*80)
        context = main_context
        for file_path, diff in file_diffs.items():
            log.info("-"*80)
            context = main_context + f" Here is the file diff for {file_path}:\n{diff} for review\n" + \
                f"You have access to the following MCP tools to help you with your code review: {tool_schemas_content}"
            response = call_llm_command.execute(context)
            # log.info the response
            log.debug(f"LLM response: {response}")
            #parse the yaml response to check if its a plan or final review
            try:
                code_blocks = extract_code_blocks(response)
                yaml_payload = code_blocks[0] if code_blocks else response
                response_data = yaml.safe_load(yaml_payload)
            except Exception as exc:
                log.error(f"Error parsing LLM response as YAML: {exc}")
                return response
            with open(f"./logs/step_{time_hash}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(response_data, f)  
            return
        #------------------------------------------------------------------
        # go throught the parsed response_data to see if its a plan or final review
        #------------------------------------------------------------------
        
        log.info("Parsed LLM response data in yaml")
        
        steps = response_data.get("steps", [])
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                log.warning("Skipping step %s because it is not a mapping: %r", index, step)
                continue
            name = step.get("name", "<unnamed>")
            description = step.get("description", "")
            log.info(f"Step {index}: {name}")
            log.info(f"Description: {description}")
                         
        #------------------------------------------------------------------
        # Execute each step in the plan
        #------------------------------------------------------------------
        def get_prompt_for_step(name, desc):
            return f"""
                You are an expert task executor.
                You are give Step: {name} to execute with details: {desc}\n
                You need to create yaml like {task_schema_content} to execute the step. 
                You have access to the following MCP tools to help you with your code review: {tool_schemas_content} 
                Make sure to follow the task schema format exactly and output only the yaml between codeblocks ``` and  ```.
                """
        
        
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                log.warning("Skipping step %s because it is not a mapping: %r", index, step)
                continue
            name = step.get("name", "<unnamed>")
            description = step.get("description", "")
            prompt =get_prompt_for_step(name, description)
        
            response = call_llm_command.execute(prompt)
            # log.info the response
            log.debug(f"LLM response: {response}")
            #parse the yaml response to check if its a plan or final review
            try:
                code_blocks = extract_code_blocks(response)
                yaml_payload = code_blocks[0] if code_blocks else response
                response_data = yaml.safe_load(yaml_payload)
            except Exception as exc:
                log.error(f"Error parsing LLM response as YAML: {exc}")
                return response
            log.info(f"Response data for step {name}: {response_data}")
        
            tasks = response_data.get("tasks", [])
            for index, step in enumerate(tasks, start=1):
                if not isinstance(step, dict):
                    log.warning("Skipping step %s because it is not a mapping: %r", index, step)
                    continue
                task_name = step.get("task_name", "<unnamed>")
                inputs = step.get("inputs", {})
                tools = step.get("tools", [])
                log.info(f"task_name {task_name}: {inputs} Tools: {tools}")
                # write the yaml back to file for debugging
                with open(f"./logs/task_{name}_{task_name}_{time_hash}.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(step, f)
                
            
            # Check if the response is a valid JSON
            # if response.startswith("TOOL_CALL:"):
            #     # Extract the JSON part
            #     response = response[len("TOOL_CALL:"):].strip()
            #     log.info(f"Extracted JSON: {response}")
            #     try:
            #         tool_call_payload = json.loads(response)
            #     except json.JSONDecodeError as exc:
            #         tool_result = f"Invalid JSON response from LLM. Error: {exc}. Original payload: {response}"
            #         isSuceess = False
            #     else:
            #         server_target = str(tool_call_payload.get("server", "ast")).lower()
            #         if server_target == "search":
            #                 command = search_tool_call_command
            #         elif server_target == "ast":
            #             command = ast_tool_call_command
            #         else:
            #             command = None
            #             tool_result = (
            #                 f"Unknown server '{server_target}'. Please set 'server' to either 'ast' or 'search'."
            #             )
            #             isSuceess = False
            #         if command:
            #             tool_result,isSuceess =await command.execute(response)
            #     log.info(f"Tool result: {tool_result}")
            #     # check before adding to context
            #     temp =context + f"Tool call result: {tool_result}"
            #     if num_tokens_from_string(temp) < MAX_CONTEXT_LENGTH-10:
            #         context = temp
            #     else:
            #         log.warning("Context too long, not adding tool result to context.")
            if "DONE" in response:
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
