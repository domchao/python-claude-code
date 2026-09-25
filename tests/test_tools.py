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

    def run(self) -> ToolResult:
        raise RuntimeError("kaboom")


@pytest.fixture
def runtime() -> AgentToolRuntime:
    return AgentToolRuntime(tools=[ReadFile, Write, Edit, Bash, Boom])


class TestRuntime:
    def test_tool_specs_use_class_names_and_docstrings(self, runtime):
        specs = {s["name"]: s for s in runtime.tool_specs}
        assert set(specs) == {"ReadFile", "Write", "Edit", "Bash", "Boom"}
        assert specs["ReadFile"]["description"] == "Read a file from a filepath."
        assert "filepath" in specs["ReadFile"]["input_schema"]["properties"]

    def test_unknown_tool(self, runtime):
        result = runtime.run_tool("Nope", {})
        assert result.result is None
        assert "Unknown tool 'Nope'" in result.error

    def test_invalid_arguments(self, runtime):
        result = runtime.run_tool("ReadFile", {})
        assert result.result is None
        assert "Invalid arguments for 'ReadFile'" in result.error

    def test_tool_exception_is_reported_not_raised(self, runtime):
        result = runtime.run_tool("Boom", {})
        assert result.result is None
        assert "kaboom" in result.error


class TestReadFile:
    def test_reads_content(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        result = runtime.run_tool("ReadFile", {"filepath": str(f)})
        assert result == ToolResult(result="hello", error=None)

    def test_missing_file(self, runtime, tmp_path: Path):
        result = runtime.run_tool("ReadFile", {"filepath": str(tmp_path / "x")})
        assert result.result is None
        assert "Failed to read" in result.error

    def test_non_utf8_file(self, runtime, tmp_path: Path):
        f = tmp_path / "bin"
        f.write_bytes(b"\xff\xfe\x00")
        result = runtime.run_tool("ReadFile", {"filepath": str(f)})
        assert result.result is None
        assert "Failed to read" in result.error


class TestWrite:
    def test_writes_and_creates_parents(self, runtime, tmp_path: Path):
        f = tmp_path / "nested" / "dir" / "a.txt"
        result = runtime.run_tool("Write", {"filepath": str(f), "content": "hi"})
        assert result.error is None
        assert f.read_text() == "hi"

    def test_overwrites(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("old")
        runtime.run_tool("Write", {"filepath": str(f), "content": "new"})
        assert f.read_text() == "new"

    def test_write_failure(self, runtime, tmp_path: Path):
        # Target is a directory, so open() fails.
        result = runtime.run_tool("Write", {"filepath": str(tmp_path), "content": "x"})
        assert result.result is None
        assert "Failed to write" in result.error


class TestEdit:
    def edit(self, runtime, path: Path, old: str, new: str, **kw) -> ToolResult:
        return runtime.run_tool(
            "Edit", {"filepath": str(path), "old_str": old, "new_str": new, **kw}
        )

    def test_single_replacement(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("foo bar")
        result = self.edit(runtime, f, "foo", "baz")
        assert result.error is None
        assert "1 edit" in result.result
        assert f.read_text() == "baz bar"

    def test_missing_file(self, runtime, tmp_path: Path):
        result = self.edit(runtime, tmp_path / "nope", "a", "b")
        assert "Path does not exist" in result.error

    def test_directory_is_rejected(self, runtime, tmp_path: Path):
        result = self.edit(runtime, tmp_path, "a", "b")
        assert "Path does not exist" in result.error

    def test_old_str_not_found(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("foo")
        result = self.edit(runtime, f, "zzz", "b")
        assert "not found" in result.error
        assert f.read_text() == "foo"

    def test_ambiguous_match_requires_replace_all(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("x x")
        result = self.edit(runtime, f, "x", "y")
        assert "multiple times" in result.error
        assert f.read_text() == "x x"

    def test_replace_all(self, runtime, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("x x x")
        result = self.edit(runtime, f, "x", "y", replace_all=True)
        assert "3 edit" in result.result
        assert f.read_text() == "y y y"


class TestBash:
    def test_captures_stdout_and_stderr(self, runtime):
        result = runtime.run_tool("Bash", {"command": "echo out; echo err >&2"})
        assert result.error is None
        assert "out" in result.result
        assert "err" in result.result

    def test_respects_working_dir(self, runtime, tmp_path: Path):
        result = runtime.run_tool(
            "Bash", {"command": "pwd", "working_dir": str(tmp_path)}
        )
        assert str(tmp_path.resolve()) in result.result

    def test_invalid_working_dir(self, runtime, tmp_path: Path):
        result = runtime.run_tool(
            "Bash", {"command": "true", "working_dir": str(tmp_path / "nope")}
        )
        assert "Invalid working directory" in result.error

    def test_non_positive_timeout(self, runtime):
        result = runtime.run_tool("Bash", {"command": "true", "timeout_seconds": 0})
        assert "timeout_seconds must be > 0" in result.error

    def test_timeout(self, runtime):
        result = runtime.run_tool("Bash", {"command": "sleep 5", "timeout_seconds": 1})
        assert result.result is None
        assert "timed out" in result.error

    def test_nonzero_exit_is_not_an_error(self, runtime):
        # Documents current behaviour: exit code is ignored (check=False).
        result = runtime.run_tool("Bash", {"command": "exit 3"})
        assert result.error is None
