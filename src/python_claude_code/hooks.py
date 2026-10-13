import difflib
from collections.abc import Awaitable
from typing import Any, Literal, Protocol

from anthropic.types import ToolUseBlock
from rich import print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from .tools import ToolResult

console = Console()

type HookType = Literal[
    "on_model_response",
    "on_text_delta",
    "on_tool_call",
    "on_tool_result",
]


class OnModelResponseHook(Protocol):
    def __call__(self, *, text: str) -> Awaitable[None]: ...


class OnTextDeltaHook(Protocol):
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


type AnyHook = OnModelResponseHook | OnTextDeltaHook | OnToolCallHook | OnToolResultHook


async def print_llm_response(*, text: str):
    if text:
        print()
        print(Markdown(f"Assistant: {text}"))
        print()


class StreamingMarkdownPrinter:
    """Renders streamed text deltas as live-updating Markdown.

    Register `on_delta` for "on_text_delta" and `on_done` for "on_model_response".
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._buffer = ""
        self._live: Live | None = None

    async def on_delta(self, *, text: str) -> None:
        if not text:
            return
        if self._live is None:
            self._live = Live(
                console=self._console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        self._buffer += text
        self._live.update(Markdown(self._buffer))

    async def on_done(self, *, text: str) -> None:
        # Tool-only turns stream no text, so there is nothing to finish.
        if self._live is None:
            return
        self._live.update(Markdown(self._buffer))
        self._live.stop()
        self._live = None
        self._buffer = ""


async def print_tool_call(*, call: ToolUseBlock) -> None:
    print(f"Tool Call: {call.name}:{call.input}")


async def print_tool_result(
    *,
    result: ToolResult,
    call_name: str,
    call_args: dict[str, Any],
):
    edit_names = {"edit"}
    if call_name.lower() in edit_names:
        old_str = str(call_args.get("old_str", ""))
        new_str = str(call_args.get("new_str", ""))
        path = str(call_args.get("filepath", "<unknown>"))
        print(f"[bold]Edit[/bold] {path}")

        if old_str or new_str:
            diff_text = "\n".join(
                difflib.unified_diff(
                    old_str.splitlines(),
                    new_str.splitlines(),
                    fromfile=f"{path} (before)",
                    tofile=f"{path} (after)",
                    lineterm="",
                )
            )
            if diff_text:
                print(
                    Panel(
                        Syntax(
                            diff_text,
                            "diff",
                            line_numbers=True,
                            word_wrap=False,
                        ),
                        title=f"Diff preview: {path}",
                        border_style="cyan",
                    )
                )
        return
    status = "[green]✓[/green]" if not result.error else "[red]✗[/red]"
    print(f"{status} [bold]{call_name}[/bold] {call_args}")
