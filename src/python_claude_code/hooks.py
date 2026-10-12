from collections.abc import Awaitable
from typing import Any, Literal, Protocol

from anthropic.types import ToolUseBlock

from .tools import ToolResult

type HookType = Literal[
    "on_model_response",
    "on_tool_call",
    "on_tool_result",
]


class OnModelResponseHook(Protocol):
    def __call__(self, *, text: str) -> Awaitable[None]: ...


class OnToolCallHook(Protocol):
    def __call__(self, *, call: ToolUseBlock) -> Awaitable[None]: ...


class OnToolResultHook(Protocol):
    def __call__(
        self,
        *,
        result: ToolResult,
        call_name: str,
        call_args: dict[str, Any],
    ) -> Awaitable[None]: ...


type AnyHook = OnModelResponseHook | OnToolCallHook | OnToolResultHook


async def print_llm_response(*, text: str) -> None:
    if text:
        print(f"\nAssistant: {text}")


async def print_tool_call(*, call: ToolUseBlock) -> None:
    print(f"Tool Call: {call.name}:{call.input}")


async def print_tool_result(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
) -> None:
    status = "✓" if not result.error else "✗"
    print(f"{status} {call_name}: {call_args}")
