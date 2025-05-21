"""
  This is much simple and based on FastMCP
  
  from https://github.com/modelcontextprotocol/python-sdk/?tab=readme-ov-file#quickstart
  and from   https://gofastmcp.com/deployment/running-server
  
"""
from fastmcp import FastMCP
from tools.code_indexer import get_function_context_for_project,find_function_calls_within_project

mcp = FastMCP()

# Assume that this is the tool you want to expose
# Give all the types and description
@mcp.tool()
def get_function_context_for_project_mcp(function_name:str, github_repo:str,)-> str:
    """
    Get the details of a function in a GitHub repo along with its callees.
    
    @param function_name: The name of the function to find.
    @param github_repo: The URL of the GitHub repo.
    @param project_root: The root directory of the project.
    """
    return get_function_context_for_project(function_name, github_repo)

@mcp.tool()
def get_function_references_mcp(function_name:str, github_repo:str,)-> str:
    """
    Get the references of a function in a GitHub repo.
    
    @param function_name: The name of the function whose references to find.
    @param github_repo: The URL of the GitHub repo.
    """
    callees = find_function_calls_within_project(function_name,github_repo)
    return callees

if __name__ == "__main__":
    try:
        mcp.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=4200,
            path="/mcp",
            log_level="debug",
        )
    except Exception as e:
        print(f"Failed to start MCP server: {e}")