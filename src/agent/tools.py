from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


ToolHandler = Callable[[dict], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: ToolHandler
    parameters_schema: dict[str, Any] | None = None


class ToolRegistry:
    """Tool registry keeps the tool layer independent from agent logic."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already exists: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, args: dict) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(args)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in self.list_tools():
            schema = tool.parameters_schema or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                }
            )
        return tools
