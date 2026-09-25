import asyncio
import contextlib
import os
import signal
from functools import cached_property
from typing import Any

from anthropic.types import ToolParam
from pydantic import BaseModel, Field, ValidationError


class ToolResult(BaseModel):
    result: str | None
    error: str | None


class AgentTool(BaseModel):
    def run_sync(self) -> ToolResult:
        """Override for blocking tools; runs in a worker thread."""
        raise NotImplementedError

    async def run(self) -> ToolResult:
        """Override directly for tools that are natively async (e.g. Bash)."""
        return await asyncio.to_thread(self.run_sync)


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

    async def run_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
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
            return await tool_fn.run()
        except Exception as exc:  # noqa: BLE001 - tool errors are reported to the model, not raised
            return ToolResult(result=None, error=f"Tool '{tool_name}' failed: {exc}")


class ReadFile(AgentTool):
    """Read a file from a filepath."""

    filepath: str = Field(description="Filepath of file to read.")

    def run_sync(self) -> ToolResult:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return ToolResult(result=f.read(), error=None)
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult(
                result=None, error=f"Failed to read '{self.filepath}': {exc}"
            )


class Write(AgentTool):
    """Write string content to a filepath"""

    filepath: str = Field(description="Filepath of file to write to.")
    content: str = Field(description="Content to write to file.")

    def run_sync(self) -> ToolResult:
        try:
            parent = os.path.dirname(self.filepath)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(self.content)

            return ToolResult(
                result=f"successfully wrote content to {self.filepath}",
                error=None,
            )
        except (OSError, UnicodeEncodeError) as e:
            return ToolResult(
                result=None,
                error=f"Failed to write '{self.filepath}': {e}",
            )


class Edit(AgentTool):
    """Replace a string in a file."""

    filepath: str = Field(description="Filepath of file to edit")
    old_str: str = Field(description="The string to replace")
    new_str: str = Field(description="The new string")
    replace_all: bool = False

    def run_sync(self) -> ToolResult:
        if not os.path.exists(self.filepath) or not os.path.isfile(self.filepath):
            return ToolResult(
                result=None,
                error=f"Path does not exist: {self.filepath}",
            )

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError) as e:
            return ToolResult(
                result=None,
                error=f"Failed to read '{self.filepath}': {e}",
            )

        if self.old_str not in original:
            return ToolResult(
                result=None,
                error=f"old_str not found in {self.filepath}",
            )

        occurrences = original.count(self.old_str)
        if not self.replace_all and occurrences > 1:
            return ToolResult(
                result=None,
                error="old_str appears multiple times; set replace_all=True or provide a more specific old_str",
            )

        if self.replace_all:
            updated = original.replace(self.old_str, self.new_str)
            replacements = occurrences
        else:
            updated = original.replace(self.old_str, self.new_str, 1)
            replacements = 1

        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.write(updated)
            return ToolResult(
                result=f"Applied {replacements} edit(s) to {self.filepath}", error=None
            )
        except (OSError, UnicodeEncodeError) as e:
            return ToolResult(
                result=None,
                error=f"Failed to write '{self.filepath}': {e}",
            )


class Bash(AgentTool):
    """Use the bash command line tool to perform computer file operations."""

    command: str = Field(description="The bash command to use.")
    working_dir: str = "."
    timeout_seconds: int = 30

    async def run(self) -> ToolResult:
        if self.timeout_seconds <= 0:
            return ToolResult(
                result=None,
                error="timeout_seconds must be > 0",
            )

        if not os.path.isdir(self.working_dir):
            return ToolResult(
                result=None,
                error=f"Invalid working directory: {self.working_dir}",
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                self.command,
                cwd=self.working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group, so we can kill children too
            )
        except OSError as e:
            return ToolResult(result=None, error=f"Bash execution failed: {e}")

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError:
            await self._kill(proc)
            return ToolResult(
                result=None, error=f"Command timed out after {self.timeout_seconds}s"
            )
        except asyncio.CancelledError:
            await self._kill(proc)
            raise

        return ToolResult(
            error=None,
            result=(
                "Executed command successfully\n\n"
                "<output>\n"
                f"{stdout.decode(errors='replace')}{stderr.decode(errors='replace')}"
                "</output>"
            ),
        )

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
