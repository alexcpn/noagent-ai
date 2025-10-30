"""Example of using the OpenAI Agents SDK to plan and execute a code review.

Run with:
    uv run python test.py path/to/code.py

If no file path is provided, a small sample snippet is reviewed instead.
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

from openai import AsyncOpenAI

from agents import Agent, Runner
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel


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
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="sk-local",  # Ollama accepts any non-empty API key string.
    )
    return OpenAIChatCompletionsModel(model="phi3.5", openai_client=ollama_client)


def plan_review(model: OpenAIChatCompletionsModel, code_snippet: str) -> list[dict]:
    """Ask a planner agent to produce structured review steps in JSON."""
    planner = Agent(
        name="Review Planner",
        instructions=dedent(
            """
            You are a principal engineer planning a code review.
            Always reply with valid JSON only.
            Use the schema: {"steps":[{"title": str, "goal": str, "prompt": str}]}
            Keep between 3 and 6 steps. Each step should focus on a distinct review concern.
            Never add commentary outside the JSON object.
            """
        ).strip(),
        model=model,
    )

    planner_prompt = dedent(
        f"""
        Create a short sequence of review steps for the following code snippet.
        Make sure each step is actionable and focused on reviewing the code.

        ```python
        {code_snippet}
        ```
        """
    ).strip()

    result = Runner.run_sync(planner, planner_prompt)
    plan_text = result.final_output.strip()

    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Planner did not return valid JSON: {plan_text}") from exc

    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"No review steps supplied: {plan_text}")

    return steps


def execute_step(
    model: OpenAIChatCompletionsModel,
    code_snippet: str,
    step_number: int,
    step: dict,
) -> str:
    """Instantiate an agent for a specific review step and run it."""
    title = step.get("title") or f"Step {step_number}"
    goal = step.get("goal") or "Review the code snippet"
    extra_prompt = step.get("prompt") or ""

    reviewer = Agent(
        name=f"Reviewer {step_number}: {title}",
        instructions=dedent(
            f"""
            You are performing code review step {step_number}: {title}.
            Focus on the goal: {goal}.
            Provide specific, actionable feedback grounded in the code.
            Keep the answer concise and use bullet points when appropriate.
            """
        ).strip(),
        model=model,
    )

    review_prompt = dedent(
        f"""
        Code snippet under review:

        ```python
        {code_snippet}
        ```

        Step goal: {goal}
        Additional guidance: {extra_prompt}

        Produce the findings for this step.
        """
    ).strip()

    result = Runner.run_sync(reviewer, review_prompt)
    return result.final_output.strip()


def main() -> None:
    code_snippet = load_code_snippet()
    model = build_model()

    steps = plan_review(model, code_snippet)
    print("Planned review steps:")
    for index, step in enumerate(steps, start=1):
        print(f"  {index}. {step.get('title', f'Step {index}')}: {step.get('goal', '')}")

    print("\nExecuting review steps...\n")
    for index, step in enumerate(steps, start=1):
        findings = execute_step(model, code_snippet, index, step)
        print(f"=== Step {index}: {step.get('title', f'Step {index}')} ===")
        print(findings)
        print()


if __name__ == "__main__":
    main()
