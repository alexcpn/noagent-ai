"""
Schema-driven agent factory.

Loads a YAML schema describing tasks and tools, then wires them together so
tasks can be executed by delegating to the referenced tool functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

import yaml


class SchemaError(RuntimeError):
    """Raised when the YAML schema is missing required information."""


TYPE_CASTERS: Mapping[str, Callable[[Any], Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


@dataclass(frozen=True)
class IOField:
    name: str
    type: str
    description: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IOField":
        try:
            return cls(
                name=payload["name"],
                type=payload["type"],
                description=payload.get("description", ""),
            )
        except KeyError as exc:
            raise SchemaError(f"IO field missing required key: {exc}") from exc

    def convert(self, value: Any) -> Any:
        caster = TYPE_CASTERS.get(self.type)
        if caster is None:
            return value
        try:
            return caster(value)
        except Exception as exc:
            raise ValueError(
                f"Unable to cast value '{value}' for field '{self.name}' to '{self.type}'"
            ) from exc


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    description: str
    inputs: List[IOField]
    outputs: List[IOField]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskDefinition":
        try:
            inputs = [IOField.from_dict(item) for item in payload.get("inputs", [])]
            outputs = [IOField.from_dict(item) for item in payload.get("outputs", [])]
            return cls(
                name=payload["name"],
                description=payload.get("description", ""),
                inputs=inputs,
                outputs=outputs,
            )
        except KeyError as exc:
            raise SchemaError(f"Task missing required key: {exc}") from exc

    def normalize_inputs(self, supplied: Mapping[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for field in self.inputs:
            if field.name not in supplied:
                raise ValueError(f"Missing required input '{field.name}' for task '{self.name}'")
            normalized[field.name] = field.convert(supplied[field.name])
        return normalized

    def normalize_outputs(self, result: Any) -> Dict[str, Any]:
        if len(self.outputs) == 0:
            return {}
        if len(self.outputs) == 1:
            field = self.outputs[0]
            if isinstance(result, Mapping) and field.name in result:
                value = result[field.name]
            else:
                value = result
            return {field.name: value}
        if not isinstance(result, Mapping):
            raise ValueError(
                f"Task '{self.name}' expected a mapping for outputs "
                f"{[field.name for field in self.outputs]} but received {type(result)}"
            )
        normalized: Dict[str, Any] = {}
        for field in self.outputs:
            if field.name not in result:
                raise ValueError(
                    f"Output '{field.name}' missing from tool result for task '{self.name}'"
                )
            normalized[field.name] = result[field.name]
        return normalized


@dataclass
class ToolFunctionDefinition:
    module_name: str | None
    name: str
    description: str
    inputs: List[IOField]
    outputs: List[IOField]
    implementation: str | None = None
    language: str | None = None
    _callable_cache: Callable[..., Any] | None = field(init=False, default=None)

    @classmethod
    def from_dict(cls, module_name: str, payload: Mapping[str, Any]) -> "ToolFunctionDefinition":
        try:
            return cls(
                module_name=module_name,
                name=payload["name"],
                description=payload.get("description", ""),
                inputs=[IOField.from_dict(item) for item in payload.get("inputs", [])],
                outputs=[IOField.from_dict(item) for item in payload.get("outputs", [])],
                implementation=payload.get("implementation"),
                language=payload.get("language"),
            )
        except KeyError as exc:
            raise SchemaError(f"Tool function missing required key: {exc}") from exc

    def build_callable(self) -> Callable[..., Any]:
        if self._callable_cache is None:
            self._callable_cache = self._create_callable()
        return self._callable_cache

    def _create_callable(self) -> Callable[..., Any]:
        if self.implementation:
            language = (self.language or "python").lower()
            if language != "python":
                raise SchemaError(
                    f"Unsupported implementation language '{self.language}' for function '{self.name}'"
                )
            namespace: Dict[str, Any] = {}
            exec(self.implementation, namespace)
            func = namespace.get(self.name)
            if not callable(func):
                raise SchemaError(
                    f"Implementation for function '{self.name}' must define a callable named '{self.name}'"
                )
            return func
        if not self.module_name:
            raise SchemaError(
                f"Function '{self.name}' must provide a module name or inline implementation"
            )
        callable_obj = getattr(import_module(self.module_name), self.name, None)
        if callable_obj is None:
            raise SchemaError(
                f"Module '{self.module_name}' does not define callable '{self.name}'"
            )
        return callable_obj


@dataclass
class TaskHandler:
    task: TaskDefinition
    function: ToolFunctionDefinition
    callable: Callable[..., Any]

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        normalized_args = self.task.normalize_inputs(kwargs)
        result = self.callable(**normalized_args)
        return self.task.normalize_outputs(result)


class SchemaAgent:
    """Agent that is assembled dynamically from a YAML task schema."""

    def __init__(self, handlers: Mapping[str, TaskHandler]) -> None:
        self._handlers = dict(handlers)

    @classmethod
    def from_schema_file(cls, path: str | Path) -> "SchemaAgent":
        schema = _load_schema(path)
        tasks = [TaskDefinition.from_dict(item) for item in schema.get("tasks", [])]
        tool_functions = _load_tool_functions(schema.get("tools", []))
        handlers = cls._wire_handlers(tasks, tool_functions)
        return cls(handlers)

    @staticmethod
    def _wire_handlers(
        tasks: Iterable[TaskDefinition],
        tool_functions: Mapping[str, ToolFunctionDefinition],
    ) -> Dict[str, TaskHandler]:
        handlers: Dict[str, TaskHandler] = {}
        for task in tasks:
            function = _match_tool_function(task, tool_functions)
            callable_obj = function.build_callable()
            handlers[task.name] = TaskHandler(task=task, function=function, callable=callable_obj)
        return handlers

    def list_tasks(self) -> List[str]:
        return list(self._handlers.keys())

    def describe_task(self, task_name: str) -> str:
        handler = self._handlers.get(task_name)
        if handler is None:
            raise KeyError(f"Task '{task_name}' not found")
        return handler.task.description

    def execute(self, task_name: str, **kwargs: Any) -> Dict[str, Any]:
        handler = self._handlers.get(task_name)
        if handler is None:
            raise KeyError(f"Task '{task_name}' not found")
        return handler(**kwargs)


def _load_schema(path: str | Path) -> Mapping[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Schema file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_tool_functions(raw_tools: Iterable[Mapping[str, Any]]) -> Dict[str, ToolFunctionDefinition]:
    functions: Dict[str, ToolFunctionDefinition] = {}
    for tool in raw_tools:
        module_name = tool.get("module") or tool.get("name")
        if not tool.get("name"):
            raise SchemaError("Tool definition missing 'name'")
        for payload in tool.get("functions", []):
            definition = ToolFunctionDefinition.from_dict(module_name, payload)
            functions[definition.name] = definition
    return functions


def _match_tool_function(
    task: TaskDefinition, tool_functions: Mapping[str, ToolFunctionDefinition]
) -> ToolFunctionDefinition:
    # Prefer direct name match first.
    direct = tool_functions.get(task.name)
    if direct:
        return direct

    # Fallback to matching on identical output fields.
    task_output_names = {field.name for field in task.outputs}
    candidates: List[ToolFunctionDefinition] = []
    for definition in tool_functions.values():
        function_output_names = {field.name for field in definition.outputs}
        if task_output_names == function_output_names and function_output_names:
            candidates.append(definition)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SchemaError(
            f"No tool function found for task '{task.name}'. "
            "Ensure the tool function name matches the task name or their outputs align."
        )
    raise SchemaError(
        f"Multiple tool functions match task '{task.name}' output signature: "
        f"{[candidate.name for candidate in candidates]}"
    )


__all__ = ["SchemaAgent", "SchemaError"]
