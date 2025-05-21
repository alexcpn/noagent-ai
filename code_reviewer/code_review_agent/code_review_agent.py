"""
Author: Alex Punnen
Status:  Demo
This is a simple python based Code Review Agent flow using OpenAI LLM APIs amd Model Context Protocl based client
Design patterns like Command Pattern are used along with for loops to stucture flow and response as we need

"""
import asyncio
from fastmcp import Client
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import requests
import re
from collections import defaultdict
import tiktoken
from abc import ABC, abstractmethod
import logging as log
from datetime import datetime

# configure logging

__author__ = "Alex Punnen"
__version__ = "1.0.0"
__email__ = "alexcpn@gmail.com"


        
#--------------------------------------------------------------------
# Helper functions
#--------------------------------------------------------------------

# Load the .env file and get the API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
MAX_CONTEXT_LENGTH = 16385
MAX_RETRIES = 5
COST_PER_TOKEN_INPUT =  0.10/10e6 # USD  # https://platform.openai.com/docs/pricing for gpt-4.1-nano
COST_PER_TOKEN_OUTPUT = .40/10e6 # USD

# Initialize OpenAI client with OpenAI's official base URL
openai_client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)

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
    

def num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo")  -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#--------------------------------------------------------------------
# Helper Classes
#--------------------------------------------------------------------

class Command(ABC):
    
    def __init__(self, client: object, description: str):
        self.client = client
        self.description = description
        
    @abstractmethod
    def execute(self, ctx: str) -> None:
        """Do something (possibly mutate ctx) or raise/return an error."""

#--------------------------------------------------------------------
       
class CallLLM(Command):
    
    #override init
    def __init__(self, client: object, description: str, model:str, cost_per_token_input: float, cost_per_token_output: float,max_budget: float = 0.5):
        """
        Initialize the CallLLM command with a client and description.
        Args:
            client (object): The client to use for making API calls.
            description (str): A description of the command.
            model (str): The model to use for the LLM
            cost_per_token (float): The cost per token for the command in dollars
            max_budget (float): The maximum budget for the command in dollars
        """
        super().__init__(client, description)
        self.client = client
        self.description = description
        self.model = model
        self.max_budget = max_budget
        self.cost_per_token_input = cost_per_token_input
        self.cost_per_token_output = cost_per_token_output
        self.max_tokens = int(self.max_budget / self.cost_per_token_input)
        self.token_limit = 4096 # max tokens for gpt-3.5-turbo
        self.tokens_inputs_used = 0
        self.tokens_outputs_generated = 0
        
    def get_total_cost(self) -> float:
        """Calculate the total cost of the command."""
        log.info(f"Total input tokens used: {self.tokens_inputs_used} Total output tokens generated: {self.tokens_outputs_generated}")
        log.info(f"Total cost: {self.tokens_inputs_used * self.cost_per_token_input + self.tokens_outputs_generated * self.cost_per_token_output} ")
        return self.tokens_inputs_used * self.cost_per_token_input + self.tokens_outputs_generated * self.cost_per_token_output
        
    def execute(self, ctx: str) -> None:
        # check if usage has exceeededed the budget
        if self.get_total_cost() > self.max_budget:
            log.warning(f"Budget exceeded Token Used {self.get_total_cost()} > max_tokens {self.max_budget}")
            return "Budget exceeded"
        
        completion = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "user", "content":ctx }
        ]
        )
        usage = completion.usage 
        log.info(f"LLM usage: {usage}")
        self.tokens_inputs_used += usage.prompt_tokens
        self.tokens_outputs_generated += usage.completion_tokens
        return completion.choices[0].message.content
            

#--------------------------------------------------------------------        

class ToolList(Command):
    async def execute(self, ctx) -> (str,bool):
            # Extract the method name and params from the context
        async with self.client as client:
            return await client.list_tools()

#--------------------------------------------------------------------        
          
class ToolCall(Command):
    async def execute(self, ctx) -> (str,bool):
            # Extract the method name and params from the context
        async with self.client as client:
            isSuccess = False
            try:
                tool_call = json.loads(ctx)
                method_name = tool_call["method"]
                params = tool_call["params"]
                isSuccess = True
                tool_result= await client.call_tool(method_name, params)
                return tool_result,isSuccess
            except json.JSONDecodeError as e:
                isSuccess = False
                log.info(f"Error decoding JSON: {e}")
                return ctx + f"Invalid JSON response from LLM: Your respose is {ctx}. This gives Error: {e} Please correct and try again",isSuccess
            except Exception as e:
                isSuccess = False
                log.info(f"Error calling tool: {e}")
                return f"Your respose is {ctx} There is an error calling tool: {e}. Please correct and try again",isSuccess
            
#--------------------------------------------------------------------

async def main():
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
    
    repo_url = "https://github.com/huggingface/accelerate"
    
    # Example: get the diff for a specific PR
    file_diffs = get_pr_diff_url(repo_url, 3321)
    
    #------------------------------------------------
    #  Command to Call the LLM with a budget ( 0.5 Dollars)
    call_llm_command = CallLLM(openai_client, "Call the LLM with the given context", "gpt-4.1-nano", COST_PER_TOKEN_INPUT,COST_PER_TOKEN_OUTPUT, 0.5)
    
    # this this the MCP client invoking the tool
    async with Client("http://127.0.0.1:4200/mcp/") as fastmcp_client:
        tool_call_command = ToolCall(fastmcp_client, "Call the tool with the given method and params")
        tool_list_command = ToolList(fastmcp_client, "List the available tools")
        
        tools = await tool_list_command.execute(None)
        log.info(f"Available tools: {tools}")
        # Example: log.info diffs for all files (trimmed)
        for file_path, diff in file_diffs.items():
            log.info("-"*80)
            log.info(f"Review diff for {file_path}") 
            
            main_context = f"You are an expert Python code reviewer, You are given the following {diff} to review from the repo {repo_url} " + \
            f"You can use the following tools {tools} if needed to get more context about the code that you are reviewing," + \
            "maybe you need to check the functions used in the code, or where they are called " + \
            f"For framing a call to the tool you can use the format of the tool '{tools}'. Frame the JSON RPC call to the tool" +  \
            "If you need to call the tool start response with TOOL_CALL:<json format for the tool call>" + \
            "here is the JSON RPC call format {{\"method\": \"<method name>\", \"params\": {{\"<param 1 name>\": {<param 1 value>}, \"<param 2 name>\": {<param 2 value>} etc }}}}" +\
            "If you have finished with the review you can start your response with 'DONE:' and give the final review comments "
            
            context = main_context  
            for i in range(MAX_RETRIES): # max 3 iterations/retires
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
                elif response.startswith("DONE:"):
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
if __name__ == "__main__":
    asyncio.run(main())