from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ToolResult:
    output: str
    error: str = ""
    truncated: bool = False

class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, sandbox_manager):
        self.sandbox_manager = sandbox_manager

    @abstractmethod
    def execute(self, sandbox_id: str, **kwargs) -> ToolResult:
        ...

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }
