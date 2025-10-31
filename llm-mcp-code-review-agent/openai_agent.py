"""Example of using the OpenAI Agents SDK to plan and execute a code review.

Run with:
    uv run python test.py path/to/code.py

If no file path is provided, a small sample snippet is reviewed instead.
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

from fastmcp import Client
from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from dotenv import load_dotenv
import git_utils as git_uitls

import os
load_dotenv()
#https://platform.openai.com/api-keys add this to your .env file



def load_code_snippet() -> str:
    """Load the code snippet specified on the command line or fall back to a demo."""
    if len(sys.argv) > 1:
        snippet_path = Path(sys.argv[1]).expanduser()
        if not snippet_path.exists():
            raise SystemExit(f"Snippet file not found: {snippet_path}")
        return snippet_path.read_text()

    return dedent(
        """
        def add_numbers(a, b):
            return a + b
        """
    ).strip()


def build_model() -> OpenAIChatCompletionsModel:
    """Configure the Agents SDK model to talk to a local Ollama endpoint."""
    # ollama_client = AsyncOpenAI(
    #     base_url="http://localhost:11434/v1",
    #     api_key="sk-local",  # Ollama accepts any non-empty API key string.
    # )
    # return OpenAIChatCompletionsModel(model="phi3.5", openai_client=ollama_client)
    # Use OpenAI official client
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = AsyncOpenAI(
          api_key=api_key,
          base_url="https://api.openai.com/v1"
    )
    return OpenAIChatCompletionsModel(model="gpt-4.1-nano", openai_client=openai_client)


@dataclass
class ReviewTask:
    """Structured representation of a single review step produced by the planner."""

    title: str
    goal: str
    prompt: str = ""

    @classmethod
    def from_payload(cls, payload: dict, fallback_title: str) -> "ReviewTask":
        title = payload.get("title") or fallback_title
        goal = payload.get("goal") or "Review the pull request diff"
        prompt = payload.get("prompt") or ""
        return cls(title=title, goal=goal, prompt=prompt)

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation of the task."""
        return {"title": self.title, "goal": self.goal, "prompt": self.prompt}


class ReviewPlanner:
    """Plan a sequence of focused review tasks for a given code snippet."""

    def __init__(
        self,
        model: OpenAIChatCompletionsModel,
        *,
        min_tasks: int = 3,
        max_tasks: int = 6,
        diff: Mapping[str, str],
        repo_url: str,
    ) -> None:
        if min_tasks < 1:
            raise ValueError("min_tasks must be at least 1")
        if max_tasks < min_tasks:
            raise ValueError("max_tasks must be greater than or equal to min_tasks")
        self.model = model
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.repo_url = repo_url
        self.diff = dict(diff)
        self.review_context = self._render_diff_for_prompt(self.diff)

        ast_server_url = os.getenv(
            "CODE_AST_MCP_SERVER_URL",
            "http://127.0.0.1:7860/mcp",
        )
        tool_call_example = '{"server": "<ast|lint>", "method": "<method name>", "params": {"<param 1 name>": <param 1 value>, "...": "..."}}'
        tool_schemas = self._load_tool_schemas(ast_server_url)

        self._planner_agent = Agent(
            name="Review Planner",
            instructions=dedent(
                f"""
                You are an expert planning a code review for the repository '{repo_url}'.
                Always reply with valid JSON only.
                Use the schema: {{"steps":[{{"title": str, "goal": str, "prompt": str}}]}}
                Keep between {self.min_tasks} and {self.max_tasks} steps. Each step should focus on a distinct review concern.
                Never add commentary outside the JSON object.

                You should generate tool calls to get more context about the code that you are reviewing if needed.
                Available tool schemas:
                {tool_schemas}
                1. **Format**: Every tool call must start with: 'TOOL_CALL:<JSON>'  where `<JSON>` is a valid JSON object matching one of the tool schemas {tool_schemas}
                2. **No extra text**: Do **not** prepend or append any other words or punctuation to the JSON.
                **Example tool call** 
                {tool_call_example}
                
                """
            ).strip(),
            model=model,
        )

    def create_tasks(self) -> list[ReviewTask]:
        """Return a list of structured review tasks for the supplied diff."""
        planner_prompt = dedent(
            f"""
            Plan a thorough review for pull request changes in '{self.repo_url}'.
            Focus on high-risk areas, cross-file impacts, and where tooling might help.

            Diffs to consider:
            {self.review_context}
            """
        ).strip()

        result = Runner.run_sync(self._planner_agent, planner_prompt)
        plan_text = result.final_output.strip()
        steps_payload = self._parse_plan(plan_text)

        tasks: list[ReviewTask] = []
        for index, step_payload in enumerate(steps_payload, start=1):
            fallback_title = f"Step {index}"
            tasks.append(ReviewTask.from_payload(step_payload, fallback_title))
        return tasks

    def _parse_plan(self, plan_text: str) -> list[dict]:
        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner did not return valid JSON: {plan_text}") from exc

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"No review steps supplied: {plan_text}")
        return steps

    @staticmethod
    def _render_diff_for_prompt(diff: Mapping[str, str], max_chars: int = 8000) -> str:
        if not diff:
            return "No diff content provided."

        sections: list[str] = []
        remaining = max_chars
        for file_path, diff_text in diff.items():
            header = f"File: {file_path}"
            snippet = diff_text.strip()
            if len(snippet) > remaining:
                snippet = snippet[: max(0, remaining - 100)] + "\n...trimmed..."
                sections.append(f"{header}\n{snippet}")
                break
            sections.append(f"{header}\n{snippet}")
            remaining -= len(snippet)
            if remaining <= 0:
                break

        return "\n\n".join(sections)

    @staticmethod
    def _normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in raw_tools or []:
            if isinstance(item, dict):
                normalized.append(item)
            elif hasattr(item, "model_dump"):
                normalized.append(item.model_dump())
            elif hasattr(item, "__dict__"):
                normalized.append(
                    {
                        key: value
                        for key, value in item.__dict__.items()
                        if not key.startswith("_")
                    }
                )
            else:
                normalized.append({"description": str(item)})
        return normalized

    @classmethod
    def _load_tool_schemas(cls, server_url: str) -> str:
        loop = asyncio.new_event_loop()
        try:
            raw_tools = loop.run_until_complete(cls._list_tools_async(server_url))
        except Exception:
            normalized: list[dict[str, Any]] = []
        else:
            normalized = cls._normalize_tools(raw_tools)
        finally:
            loop.close()
            cls._ensure_default_event_loop()
        return json.dumps({"ast": normalized}, indent=2)

    @staticmethod
    async def _list_tools_async(server_url: str) -> Any:
        async with Client(server_url) as client:
            return await client.list_tools()

    @staticmethod
    def _ensure_default_event_loop() -> None:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())


def plan_review(
    model: OpenAIChatCompletionsModel,
    diff: Mapping[str, str],
    repo_url: str,
    *,
    min_tasks: int = 3,
    max_tasks: int = 6,
) -> list[dict]:
    """Helper that returns planner output in dict form."""
    planner = ReviewPlanner(
        model,
        min_tasks=min_tasks,
        max_tasks=max_tasks,
        diff=diff,
        repo_url=repo_url,
    )
    return [task.as_dict() for task in planner.create_tasks()]


def execute_step(
    model: OpenAIChatCompletionsModel,
    review_context: str,
    step_number: int,
    step: ReviewTask,
) -> str:
    """Instantiate an agent for a specific review step and run it."""
    title = step.title or f"Step {step_number}"
    goal = step.goal or "Review the pull request diff"
    extra_prompt = step.prompt or ""

    reviewer = Agent(
        name=f"Reviewer {step_number}: {title}",
        instructions=dedent(
            f"""
            You are performing code review step {step_number}: {title}.
            Focus on the goal: {goal}.
            Provide specific, actionable feedback grounded in the diff.
            Keep the answer concise and use bullet points when appropriate.
            """
        ).strip(),
        model=model,
    )

    review_prompt = dedent(
        f"""
        Pull request diff under review:

        ```diff
        {review_context}
        ```

        Step goal: {goal}
        Additional guidance: {extra_prompt}

        Produce the findings for this step.
        """
    ).strip()

    result = Runner.run_sync(reviewer, review_prompt)
    return result.final_output.strip()


def main() -> None:
    # Example: get the diff for a specific PR
    repo_url = "https://github.com/huggingface/accelerate"
    pr_number = "2214"
    print(f"Fetching diffs for PR #{pr_number} from {repo_url}...")
    file_diffs = git_uitls.get_pr_diff_url(repo_url, pr_number)
    print(f"Fetched diffs for {len(file_diffs)} files in PR #{pr_number} from {repo_url}")
    
    model = build_model()

    planner = ReviewPlanner(model, diff=file_diffs, repo_url=repo_url)
    tasks = planner.create_tasks()
    print("Planned review steps:")
    for index, task in enumerate(tasks, start=1):
        print(f"  {index}. {task.title}: {task.goal}")

    review_context = planner.review_context

    print("\nExecuting review steps...\n")
    for index, task in enumerate(tasks, start=1):
        findings = execute_step(model, review_context, index, task)
        print(f"=== Step {index}: {task.title} ===")
        print(findings)
        print()


if __name__ == "__main__":
    main()
