"""Tiny helper script to sanity check a local vLLM/OpenAI compatible server."""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

from openai import OpenAI

DEFAULT_API_KEY = os.getenv("VLLM_OPENAI_API_KEY", "sk-local")
DEFAULT_BASE_URL = os.getenv("VLLM_OPENAI_BASE_URL", "http://localhost:8080/v1")
DEFAULT_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "phi3.5")

openai_client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)


def run_chat_completion(
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """Send a single chat.completions request and return the OpenAI response object."""
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response


def pretty_print_response(response) -> None:
    """Print the assistant message and basic usage stats."""
    choice = response.choices[0]
    content = choice.message.content or ""
    print("=== Assistant Response ===")
    print(content.strip())
    if response.usage:
        usage = response.usage
        print("\n=== Token Usage ===")
        print(
            f"prompt: {usage.prompt_tokens}, "
            f"completion: {usage.completion_tokens}, "
            f"total: {usage.total_tokens}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hit the local vLLM server with a single chat completion request."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello from the local phi3.5 server.",
        help="User message to send to the model.",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant.",
        help="Optional system prompt to steer the model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Model name to request (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature to request (default: 0.2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens (default: 256).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump the raw JSON response instead of a friendly summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    response = run_chat_completion(
        prompt=args.prompt,
        system_prompt=args.system,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
    else:
        pretty_print_response(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#python3 llm-mcp-code-review-agent/test_vllm.py "Give me a limerick about MCP"