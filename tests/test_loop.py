import asyncio
import time
from types import SimpleNamespace
from typing import Any

from python_claude_code.loop import run_loop
from python_claude_code.tools import AgentTool, AgentToolRuntime, ToolResult


class Echo(AgentTool):
    """Echo text back."""

    text: str

    def run_sync(self) -> ToolResult:
        return ToolResult(result=f"echo:{self.text}", error=None)


class Fail(AgentTool):
    """Always returns an error."""

    def run_sync(self) -> ToolResult:
        return ToolResult(result=None, error="nope")


class Sleepy(AgentTool):
    """Natively async tool that takes `delay` seconds."""

    delay: float

    async def run(self) -> ToolResult:
        await asyncio.sleep(self.delay)
        return ToolResult(result=f"slept:{self.delay}", error=None)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_block(id: str, name: str, input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


class FakeClient:
    """Stands in for anthropic.AsyncAnthropic; returns a canned response and records calls."""

    def __init__(self, content: list[SimpleNamespace], stop_reason: str):
        self.calls: list[dict[str, Any]] = []
        self._response = SimpleNamespace(content=content, stop_reason=stop_reason)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._response


runtime = AgentToolRuntime(tools=[Echo, Fail, Sleepy])


async def test_plain_text_response():
    client = FakeClient([text_block("hi"), text_block("there")], "end_turn")
    messages = [{"role": "user", "content": "hello"}]

    result = await run_loop(client, "test-model", messages, runtime)

    assert result.text == "hi\nthere"
    assert result.tool_results is None
    assert result.assistant_message["role"] == "assistant"
    assert result.assistant_message["content"] is client._response.content


async def test_request_passes_model_messages_and_tools():
    client = FakeClient([text_block("hi")], "end_turn")
    messages = [{"role": "user", "content": "hello"}]

    await run_loop(client, "test-model", messages, runtime)

    (call,) = client.calls
    assert call["model"] == "test-model"
    assert call["messages"] is messages
    assert call["tools"] == runtime.tool_specs


async def test_tool_use_runs_tools_and_returns_results():
    client = FakeClient(
        [
            text_block("calling"),
            tool_block("t1", "Echo", {"text": "a"}),
            tool_block("t2", "Echo", {"text": "b"}),
        ],
        "tool_use",
    )

    result = await run_loop(client, "m", [], runtime)

    assert result.text == "calling"
    assert result.tool_results == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "echo:a"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "echo:b"},
    ]


async def test_tool_errors_are_returned_as_content():
    client = FakeClient(
        [
            tool_block("t1", "Fail", {}),
            tool_block("t2", "Missing", {}),
        ],
        "tool_use",
    )

    result = await run_loop(client, "m", [], runtime)

    assert result.tool_results[0]["content"] == "nope"
    assert "Unknown tool 'Missing'" in result.tool_results[1]["content"]


async def test_tool_use_stop_reason_without_tool_blocks():
    client = FakeClient([text_block("odd")], "tool_use")

    result = await run_loop(client, "m", [], runtime)

    assert result.tool_results == []


async def test_tool_calls_run_in_parallel_and_keep_block_order():
    # The first call is the slowest, so it finishes last but must be listed first.
    client = FakeClient(
        [
            tool_block("t1", "Sleepy", {"delay": 0.3}),
            tool_block("t2", "Sleepy", {"delay": 0.2}),
            tool_block("t3", "Sleepy", {"delay": 0.1}),
        ],
        "tool_use",
    )

    start = time.monotonic()
    result = await run_loop(client, "m", [], runtime)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # ~0.3 in parallel, 0.6 if sequential
    assert [r["tool_use_id"] for r in result.tool_results] == ["t1", "t2", "t3"]
    assert [r["content"] for r in result.tool_results] == [
        "slept:0.3",
        "slept:0.2",
        "slept:0.1",
    ]
