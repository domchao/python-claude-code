from functools import cached_property
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field, ValidationError


class ToolResult(BaseModel):
    result: str | None
    error: str | None


class AgentTool(BaseModel):
    def run(self) -> ToolResult:
        """Override this in subclasses to define tool logic."""
        raise NotImplementedError


class AgentToolRuntime(BaseModel):
    """Minimal runtime that exposes tools to the model."""

    tools: list[type[AgentTool]] = Field(description="List of tools.")

    @cached_property
    def tool_registry(self) -> dict[str, type[AgentTool]]:
        return {t.__name__: t for t in self.tools}

    @cached_property
    def tool_specs(self) -> list[ToolParam]:
        return [self.to_spec(t) for t in self.tools]

    def to_spec(self, model: type[BaseModel]) -> ToolParam:
        return {
            "name": model.__name__,
            "description": model.__doc__ or "",
            "input_schema": model.model_json_schema(),
        }

    def run_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        tool_cls = self.tool_registry.get(tool_name)
        if tool_cls is None:
            return ToolResult(result=None, error=f"Unknown tool '{tool_name}'")
        try:
            tool_fn = tool_cls.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                result=None, error=f"Invalid arguments for '{tool_name}': {exc}"
            )
        try:
            return tool_fn.run()
        except Exception as exc:  # noqa: BLE001 - tool errors are reported to the model, not raised
            return ToolResult(result=None, error=f"Tool '{tool_name}' failed: {exc}")


class ReadFile(AgentTool):
    """Read a file from a filepath."""

    filepath: str = Field(description="Filepath of file to read.")

    def run(self) -> ToolResult:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return ToolResult(result=f.read(), error=None)
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(
                result=None, error=f"Failed to read '{self.filepath}': {exc}"
            )
