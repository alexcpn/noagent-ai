"""
 Simple Command Pattern implementation for LLM calls and tool interactions.
    This module defines a base Command class and specific command implementations
    such as CallLLM, ToolList, and ToolCall. It also includes utility functions
    for token counting and error handling.
    
"""

import json
import tiktoken
from abc import ABC, abstractmethod
import logging as log

#--------------------------------------------------------------------
# Helper Classes
#--------------------------------------------------------------------


def num_tokens_from_string(string: str, encoding_name: str = "gpt-3.5-turbo")  -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

#--------------------------------------------------------------------
#  Command Classes
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