import asyncio
import threading
import time
from pathlib import Path

import pytest

from python_claude_code.tools import (
    AgentTool,
    AgentToolRuntime,
    Bash,
    Edit,
    ReadFile,
    ToolResult,
    Write,
)


class Boom(AgentTool):
    """Always raises."""

    def run_sync(self) -> ToolResult:
        raise RuntimeError("kaboom")


class WhichThread(AgentTool):
    """Reports the thread its run_sync executes on."""

    def run_sync(self) -> ToolResult:
        return ToolResult(result=threading.current_thread().name, error=None)


class SlowSync(AgentTool):
    """Blocks its thread for a while."""

    def run_sync(self) -> ToolResult:
        time.sleep(0.3)
        return ToolResult(result="done", error=None)


class NativeAsync(AgentTool):
    """Overrides run directly; run_sync must never be called."""

    def run_sync(self) -> ToolResult:
        raise AssertionError("run_sync should not be used")

    async def run(self) -> ToolResult:
        await asyncio.sleep(0)
        return ToolResult(result="native", error=None)


class NoImpl(AgentTool):
    """Implements neither run nor run_sync."""


@pytest.fixture
def runtime() -> AgentToolRuntime:
    return AgentToolRuntime(
        tools=[
            ReadFile,
            Write,
            Edit,
            Bash,
            Boom,
            WhichThread,
            SlowSync,
            NativeAsync,
            NoImpl,
        ]
    )


class TestRuntime:
    async def test_tool_specs_use_class_names_and_docstrings(self):
        runtime = AgentToolRuntime(tools=[ReadFile, Write, Edit, Bash, Boom])
        specs = {s["name"]: s for s in runtime.tool_specs}
        assert set(specs) == {"ReadFile", "Write", "Edit", "Bash", "Boom"}
        assert specs["ReadFile"]["description"] == "Read a file from a filepath."
        assert "filepath" in specs["ReadFile"]["input_schema"]["properties"]

    async def test_unknown_tool(self, runtime):
        result = await runtime.run_tool("Nope", {})
        assert result.result is None
        assert "Unknown tool 'Nope'" in result.error

    async def test_invalid_arguments(self, runtime):
        result = await runtime.run_tool("ReadFile", {})
        assert result.result is None
        assert "Invalid arguments for 'ReadFile'" in result.error

    async def test_tool_exception_is_reported_not_raised(self, runtime):
        result = await runtime.run_tool("Boom", {})
        assert result.result is None
        assert "kaboom" in result.error

    async def test_unimplemented_tool_is_reported(self, runtime):
        result = await runtime.run_tool("NoImpl", {})
        assert result.result is None
        assert "NoImpl" in result.error


class TestAsyncRun:
    async def test_run_sync_executes_in_worker_thread(self, runtime):
        result = await runtime.run_tool("WhichThread", {})
        assert result.result != threading.current_thread().name

    async def test_run_override_bypasses_run_sync(self, runtime):
        result = await runtime.run_tool("NativeAsync", {})
        assert result == ToolResult(result="native", error=None)

    async def test_blocking_tool_does_not_block_event_loop(self, runtime):
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await runtime.run_tool("SlowSync", {})
        task.cancel()
        assert ticks >= 10  # ~30 expected; would be 0 if the loop were blocked

    async def test_blocking_tools_run_concurrently(self, runtime):
        start = time.monotonic()
        results = await asyncio.gather(
            runtime.run_tool("SlowSync", {}), runtime.run_tool("SlowSync", {})
        )
        assert time.monotonic() - start < 0.55  # ~0.3 concurrent, 0.6 serial
        assert [r.result for r in results] == ["done", "done"]


class TestReadFile:
    async def test_reads_content(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        result = await runtime.run_tool("ReadFile", {"filepath": str(f)})
        assert result == ToolResult(result="hello", error=None)

    async def test_missing_file(self, runtime, tmp_path: Path):
        result = await runtime.run_tool("ReadFile", {"filepath": str(tmp_path / "x")})
        assert result.result is None
        assert "Failed to read" in result.error

    async def test_non_utf8_file(self, runtime, tmp_path: Path):
        f = tmp_path / "bin"
        f.write_bytes(b"\xff\xfe\x00")
        result = await runtime.run_tool("ReadFile", {"filepath": str(f)})
        assert result.result is None
        assert "Failed to read" in result.error


class TestWrite:
    async def test_writes_and_creates_parents(self, runtime, tmp_path: Path):
        f = tmp_path / "nested" / "dir" / "a.txt"
        result = await runtime.run_tool("Write", {"filepath": str(f), "content": "hi"})
        assert result.error is None
        assert f.read_text() == "hi"

    async def test_overwrites(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("old")
        await runtime.run_tool("Write", {"filepath": str(f), "content": "new"})
        assert f.read_text() == "new"

    async def test_write_failure(self, runtime, tmp_path: Path):
        # Target is a directory, so open() fails.
        result = await runtime.run_tool(
            "Write", {"filepath": str(tmp_path), "content": "x"}
        )
        assert result.result is None
        assert "Failed to write" in result.error


class TestEdit:
    async def edit(self, runtime, path: Path, old: str, new: str, **kw) -> ToolResult:
        return await runtime.run_tool(
            "Edit", {"filepath": str(path), "old_str": old, "new_str": new, **kw}
        )

    async def test_single_replacement(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("foo bar")
        result = await self.edit(runtime, f, "foo", "baz")
        assert result.error is None
        assert "1 edit" in result.result
        assert f.read_text() == "baz bar"

    async def test_missing_file(self, runtime, tmp_path: Path):
        result = await self.edit(runtime, tmp_path / "nope", "a", "b")
        assert "Path does not exist" in result.error

    async def test_directory_is_rejected(self, runtime, tmp_path: Path):
        result = await self.edit(runtime, tmp_path, "a", "b")
        assert "Path does not exist" in result.error

    async def test_old_str_not_found(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("foo")
        result = await self.edit(runtime, f, "zzz", "b")
        assert "not found" in result.error
        assert f.read_text() == "foo"

    async def test_ambiguous_match_requires_replace_all(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("x x")
        result = await self.edit(runtime, f, "x", "y")
        assert "multiple times" in result.error
        assert f.read_text() == "x x"

    async def test_replace_all(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("x x x")
        result = await self.edit(runtime, f, "x", "y", replace_all=True)
        assert "3 edit" in result.result
        assert f.read_text() == "y y y"


class TestBash:
    async def test_captures_stdout_and_stderr(self, runtime):
        result = await runtime.run_tool("Bash", {"command": "echo out; echo err >&2"})
        assert result.error is None
        assert "out" in result.result
        assert "err" in result.result

    async def test_respects_working_dir(self, runtime, tmp_path: Path):
        result = await runtime.run_tool(
            "Bash", {"command": "pwd", "working_dir": str(tmp_path)}
        )
        assert str(tmp_path.resolve()) in result.result

    async def test_invalid_working_dir(self, runtime, tmp_path: Path):
        result = await runtime.run_tool(
            "Bash", {"command": "true", "working_dir": str(tmp_path / "nope")}
        )
        assert "Invalid working directory" in result.error

    async def test_non_positive_timeout(self, runtime):
        result = await runtime.run_tool(
            "Bash", {"command": "true", "timeout_seconds": 0}
        )
        assert "timeout_seconds must be > 0" in result.error

    async def test_timeout(self, runtime):
        result = await runtime.run_tool(
            "Bash", {"command": "sleep 5", "timeout_seconds": 1}
        )
        assert result.result is None
        assert "timed out" in result.error

    async def test_nonzero_exit_is_not_an_error(self, runtime):
        # Documents current behaviour: exit code is ignored (check=False).
        result = await runtime.run_tool("Bash", {"command": "exit 3"})
        assert result.error is None

    async def test_timeout_kills_background_children(self, runtime, tmp_path: Path):
        marker = tmp_path / "survived"
        command = f"(sleep 2; touch {marker}) & wait"
        result = await runtime.run_tool(
            "Bash", {"command": command, "timeout_seconds": 1}
        )
        assert "timed out" in result.error
        await asyncio.sleep(2)
        assert not marker.exists()
