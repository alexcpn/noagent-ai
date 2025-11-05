"""Simple CLI to execute schema-defined tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from schema_agent import SchemaAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a task defined in task_schema.yaml")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("task_schema.yaml"),
        help="Path to the YAML schema file",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Name of the task to execute",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Input parameter key/value pair. Repeat for multiple params.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the task result as JSON",
    )
    return parser.parse_args()


def parse_params(pairs: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not value:
            raise ValueError(f"Invalid param '{pair}'. Expected KEY=VALUE format.")
        params[key] = value
    return params


def main() -> None:
    args = parse_args()
    params = parse_params(args.param)
    agent = SchemaAgent.from_schema_file(args.schema)
    result = agent.execute(args.task, **params)
    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    # python3 schema_agent_demo.py --task fetch_pr_diffs --param repo_url=https://github.com/huggingface/accelerate --param pr_number=3321 --pretty
     main()

