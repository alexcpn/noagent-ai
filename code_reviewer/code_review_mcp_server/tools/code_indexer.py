"""
Author: Alex Punnen
Code to create for a Code Reivew tool helper for  MCP server
License: Proprietary
"""

import os, textwrap
from tree_sitter_languages import  get_language
from tree_sitter import Parser
import tempfile
from git import Repo
from enum import Enum


parser   = Parser()


all_refs = {}  # store all classes and functions in a dict
code_ref ={} # hold the code bytes

def _collect_files(root_dir, extensions=None):
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if extensions is None or any(fname.endswith(ext) for ext in extensions):
                all_files.append(os.path.join(dirpath, fname))
    return all_files


# ---------------------------------------------------------------------------
# Language queries from https://github.com/sankalp1999/code_qa/blob/fe6ce9d852aa1c371c299db22978012df4b354a0/treesitter.py#L16
# ---------------------------------------------------------------------------


class LanguageEnum(Enum):
    UNKNOWN = "unknown"
    
    PYTHON= {
        "class": """
            (class_definition
            name: (identifier) @class.name)
        """,
        "func": """
            (function_definition
            name: (identifier) @function.name)
        """,
        "doc": """
            (expression_statement (string) @docstring)
        """,
    }
    
    GO = {
    # Match struct type declarations:
    #   type Foo struct { … }
    'struct_query': r"""
        (type_spec
            name: (type_identifier) @struct.name
            type: (struct_type))
    """,

    # Match both top-level functions and methods:
    #   func Bar(...) { … }
    #   func (r Receiver) Baz(...) { … }
    'func_query': r"""
        [
            (function_declaration
                name: (identifier) @func.name)
            (method_declaration
                name: (field_identifier) @method.name)
        ]
    """,

    # Capture all comments (line or block) for docstrings:
    #   // comment
    #   /* comment */
    'doc_query': r"""
        (comment) @comment
    """
}
# ---------------------------------------------------------------------------
#  run the query and grab the captures
# ---------------------------------------------------------------------------
def _run_query(code_bytes,q_src, tag,language):
    """
    Return a list of dicts: {node, name, start_line, end_line}
    for every capture whose capture-name == tag.
    """
    if language == "python":
        lang = get_language("python")
        parser.set_language(lang)
    elif language == "go":
        lang = get_language("go")
        parser.set_language(lang)
    else:       
        raise ValueError(f"Unsupported language: {language}")
    query     = lang.query(q_src)
    root      = parser.parse(code_bytes).root_node
    captures  = query.captures(root)                 # [(node, capture_name), …]

    items = []
    for node, cap_name in captures:
        if cap_name != tag:
            continue
        name  = code_bytes[node.start_byte: node.end_byte].decode()
        block = node.parent                          # whole class/func node
        #  Get the enclosing class (if any)
        class_name = _get_enclosing_class_name(block, code_bytes)
        
        items.append({
            "node"       : block,
            "name"       : name,
            "start_line" : block.start_point[0] + 1, # 0-based → 1-based
            "end_line"   : block.end_point[0] + 1,
            "class"      : class_name  # optional, can be None
        })
    return items


# ---------------------------------------------------------------------------
#  attach the first doc-string that falls *inside* each block
# ---------------------------------------------------------------------------
def _attach_docstrings(code_bytes,items,docs):
    for itm in items:
        for d in docs:
            if (d["node"].start_byte   >= itm["node"].start_byte and
                d["node"].end_byte     <= itm["node"].end_byte):
                itm["doc"] = code_bytes[
                    d["node"].start_byte : d["node"].end_byte
                ].decode().strip('"\''" \n")
                break
        else:
            itm["doc"] = None
    return items
# ---------------------------------------------------------------------------
# Attach file name to each item
# ---------------------------------------------------------------------------
def _attach_file_name(items, file_path):
    for itm in items:
        itm["file"] = file_path
    return items

# ---------------------------------------------------------------------------
# Build a query that finds all calls to `target_name`
# ---------------------------------------------------------------------------
def _build_call_query(target_name: str):
    # 1) Bare calls:   (call function: (identifier) @call.name)
    # 2) Attr calls:   (call function: (attribute object: _ attribute: (identifier) @call.name))
    # We capture the entire call node as @call.node so we can get its position.
    # find if this is a python or go file
    if target_name.startswith("python."):
        
        return get_language("python").query(f"""
        (
        (call
            function: (identifier) @call.name
            arguments: (_)*
        ) @call.node
        (#eq? @call.name "{target_name}")
        )
        (
        (call
            function: (attribute
                        object: (_)
                        attribute: (identifier) @call.name
                    )
            arguments: (_)*
        ) @call.node
        (#eq? @call.name "{target_name}")
        )
        """)
    elif target_name.startswith("go."):
        return get_language("go").query(f"""
        (
            (call_expression
                function: (identifier) @call.name
                arguments: (_)*
            ) @call.node
            (#eq? @call.name "{target_name}")
        )
        (
            (call_expression
                function: (selector_expression
                    operand: (_)
        
        """)

def _get_enclosing_function(node, code_bytes):
    """
    Walk up from `node` until we find a function_definition.
    Return its name (string) or None if at top-level.
    """
    cur = node
    while cur:
        if cur.type == "function_definition":
            # child_by_field_name works if the grammar labels the name field
            name_node = cur.child_by_field_name("name")
            if name_node:
                return code_bytes[name_node.start_byte:name_node.end_byte].decode()
        cur = cur.parent
    return None
def _get_enclosing_class_name(node, code_bytes):
    """
    Traverse up the tree to find the enclosing class, if any.
    """
    cur = node
    while cur:
        if cur.type == "class_definition":
            name_node = cur.child_by_field_name("name")
            if name_node:
                return code_bytes[name_node.start_byte:name_node.end_byte].decode()
        cur = cur.parent
    return None

def find_call_sites(code_bytes: bytes, target_name: str):
    lang    = get_language("python")
    parser  = Parser(); parser.set_language(lang)
    query   = _build_call_query(target_name)
    tree    = parser.parse(code_bytes)
    if query:
        caps    = query.captures(tree.root_node)
    else:
        return []

    sites = []
    for node, cap in caps:
        if cap != "call.node":
            continue

        # find caller
        caller = _get_enclosing_function(node, code_bytes) or "<module>"

        # grab the raw snippet
        raw     = code_bytes[node.start_byte:node.end_byte].decode(errors="ignore")
        snippet = " ".join(raw.split())

        start_ln, end_ln = node.start_point[0]+1, node.end_point[0]+1
        sites.append({
            "caller":     caller,
            "start_line": start_ln,
            "end_line":   end_ln,
            "snippet":    snippet,
        })

    return sites

def index_all_files(project_root,git_repo_url):
    all_classes = []
    all_functions = []
    all_files = _collect_files(project_root, [".py",".go"])
    for path in all_files:
        with open(path, "r", encoding="utf8") as f:
            code = f.read()
            code_bytes    = code.encode()
            print(f"Processing {path}")
            language = LanguageEnum.UNKNOWN.value
            if path.endswith(".py"):
                language = LanguageEnum.PYTHON.value
                classes   = _run_query(code_bytes,language["class"], "class.name","python")
                functions = _run_query(code_bytes,language["func"],  "function.name","python")
                docs      = _run_query(code_bytes,language["doc"],   "docstring","python")   # optional
                classes   = _attach_docstrings(code_bytes,classes,docs)
                functions = _attach_docstrings(code_bytes,functions,docs)
                # get the file name and previous directory
                # get only file name and relative path
                file_name = os.path.basename(path)
                rel_path      = os.path.relpath(path, project_root)
                print(f"Processing {file_name} ({len(classes)} classes, {len(functions)} functions), {rel_path})")
                classes   = _attach_file_name(classes, rel_path)
                functions = _attach_file_name(functions, rel_path)
                code_ref[git_repo_url+rel_path] =code_bytes
                all_classes.extend(classes)
                all_functions.extend(functions)
            elif path.endswith(".go"):
                language = LanguageEnum.GO.value
                # fill for Go language
                structs   = _run_query(code_bytes,language["struct_query"], "struct.name","go")
                functions = _run_query(code_bytes,language["func_query"],  "func.name","go")
                docs      = _run_query(code_bytes,language["doc_query"],   "comment","go")
                structs   = _attach_docstrings(code_bytes,structs,docs)
                functions = _attach_docstrings(code_bytes,functions,docs)
                # get the file name and previous directory
                file_name = os.path.basename(path)
                rel_path      = os.path.relpath(path, project_root)
                print(f"Processing {file_name} ({len(structs)} structs, {len(functions)} functions), {rel_path})")
                structs   = _attach_file_name(structs, rel_path)
                functions = _attach_file_name(functions, rel_path)
                code_ref[git_repo_url+rel_path] =code_bytes
                all_classes.extend(structs)
                all_functions.extend(functions)
            else:
                print(f"Skipping {path}, unsupported file type.")
                continue
          
    return all_classes, all_functions
    
def get_function_context(target_name,all_functions,github_url):
    """
    Find all functions with the same name as `target_name`.
    Return their context (docstring, source code).
    
    @param target_name: The name of the function to find.
    @param all_functions: The list of all functions in the project.
    """
    matches     = [fn for fn in all_functions if fn["name"] == target_name]
    print(f"\n\nFound {len(matches)} matches for '{target_name}':")
    for fn in matches:
        start, end = fn["node"].start_byte, fn["node"].end_byte
        file_name  = fn["file"]
        code_bytes    = code_ref[github_url+file_name]
        raw_src    = code_bytes[start:end].decode()
        src        = textwrap.dedent(raw_src).rstrip()
        rel_path      = file_name
        contex = f"Definition in {rel_path} (L{fn['start_line']}–{fn['end_line']}):\n"
        if fn.get("class"):
            contex += f"{fn['class']}.{fn['name']}  (L{fn['start_line']}–{fn['end_line']})"
        else:
            contex += f"{fn['name']}  (L{fn['start_line']}–{fn['end_line']})"
            
        if fn.get("doc"):
            contex += f"\n docstring: {fn['doc']}"
        else:
            contex += "\nNo docstring found"
        contex += "\n" +src
        return contex
        
def get_code_bytes(github_repo, file_name, start_bytes, end_bytes):
    """
    Get the code bytes for a specific file and byte range.
    """
    if github_repo+file_name not in code_ref:
        return (f"File {file_name} not found in code_ref.")
    # get the code bytes for the file
    code_bytes = code_ref[github_repo+file_name]
    # get the code bytes for the lines
    code_bytes = code_bytes[start_bytes:end_bytes]
    return code_bytes

# find all calls to a specific function in the
def find_function_calls_within_project(function_name,github_repo):
    """
    Find all calls to `target_name` in the project.
    """
    contexts = " "
    # get all keys of dict code_ref
    all_files = code_ref.keys()
    for name in all_files:
        if name.startswith(github_repo):
            code_bytes = code_ref[name]
            calls = find_call_sites(code_bytes, function_name)
            rel_path = name
            if calls:
                context = f"\nFound {len(calls)} call(s) to `{function_name}` in {rel_path}:"
                for c in calls:
                    context += f"\n  ─ in `{c['caller']}` (L{c['start_line']}–L{c['end_line']}): {c['snippet']}"
                    contexts += context
    if contexts == " ":
        contexts = f"\nNo calls to `{function_name}` found in the project."
    return contexts

def get_function_context_for_project(function_name:str, github_repo:str,)-> str:
    """
    Get the details of a function in a GitHub repo along with its callees.
    
    @param function_name: The name of the function to find.
    @param github_repo: The URL of the GitHub repo.
    @param project_root: The root directory of the project.
    """
    try:
        
        if  github_repo in all_refs:
            all_classes = all_refs[github_repo]["classes"]
            all_functions = all_refs[github_repo]["functions"]
        else:
            # Create temporary directory
            temp_dir = tempfile.TemporaryDirectory()
            project_root = temp_dir.name
            # Clone the repo
            Repo.clone_from(github_repo, project_root,depth=1)
            all_classes, all_functions =index_all_files(project_root,github_repo)
            # store this in a dict
            all_refs[github_repo] = {"classes": all_classes, "functions": all_functions}
        # store this in a dict
        contex = get_function_context(function_name,all_functions,github_repo)
        return contex
    except Exception as e:
        return f"Error: {e}"
    


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    #  For testing purposes, we can use a local directory or a GitHub repo URL.
    # ---------------------------------------------------------------------------

    #  Test with a GitHub repo URL - Pyhon repo

    print("-----------------Python Repo---------------------------------------")
    repo_url = 'https://github.com/huggingface/accelerate'
    # find a specific function
    target_name = "get_max_layer_size"
    contex =get_function_context_for_project(target_name,repo_url)
    print(contex)
    target_name = "get_max_layer_size"
    contex =get_function_context_for_project(target_name,repo_url)
    print(contex)
    print("------------------End Test Python Repo--------------------------------------")

    
    print("-----------------Go Repo---------------------------------------")
    repo_url = 'https://github.com/ngrok/ngrok-operator'
    # find a specific function
    target_name = "createKubernetesOperator"
    contex =get_function_context_for_project(target_name,repo_url)
    print(contex)
    target_name = "createKubernetesOperator"
    contex =get_function_context_for_project(target_name,repo_url)
    print(contex)
    
    #  Test with a GitHub repo URL - Pyhon repo
 

  