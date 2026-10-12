from collections.abc import Awaitable
from typing import Any, Literal, Protocol

from anthropic.types import MessageParam, ToolUseBlock

from .tools import ToolResult

type HookType = Literal[
    "on_model_response",
    "on_tool_call",
    "on_tool_result",
]


class OnModelResponseHook(Protocol):
    def __call__(self, *, message: MessageParam) -> Awaitable[None]: ...


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
