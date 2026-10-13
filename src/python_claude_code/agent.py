import asyncio
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock

from .hooks import (
    AnyHook,
    HookType,
    OnModelResponseHook,
    OnTextDeltaHook,
    OnToolCallHook,
    OnToolResultHook,
)
from .tools import AgentToolRuntime


@dataclass
class ToolLoopResult:
    assistant_message: MessageParam
    text: str = ""
    tool_results: list[ToolResultBlockParam] | None = None


class Agent:
    def __init__(
        self,
        tool_runtime: AgentToolRuntime,
        model: str = "claude-haiku-4-5-20251001",
        client: AsyncAnthropic | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self.tool_runtime = tool_runtime
        self.model = model
        self.client = client if client is not None else AsyncAnthropic()
        self.max_tokens = max_tokens
        self._hooks: dict[HookType, list[AnyHook]] = {
            "on_model_response": [],
            "on_text_delta": [],
            "on_tool_call": [],
            "on_tool_result": [],
        }

    @overload
    def on(
        self, event: Literal["on_model_response"], handler: OnModelResponseHook
    ) -> "Agent": ...

    @overload
    def on(
        self, event: Literal["on_text_delta"], handler: OnTextDeltaHook
    ) -> "Agent": ...

    @overload
    def on(
        self, event: Literal["on_tool_call"], handler: OnToolCallHook
    ) -> "Agent": ...

    @overload
    def on(
        self, event: Literal["on_tool_result"], handler: OnToolResultHook
    ) -> "Agent": ...

    def on(self, event: HookType, handler: AnyHook) -> "Agent":
        self._hooks[event].append(handler)
        return self

    async def emit(self, event: HookType, **kwargs: Any) -> None:
        for handler in self._hooks[event]:
            try:
                await handler(**kwargs)
            except Exception as exc:  # noqa: BLE001 - isolate hook failures so one broken handler can't take down the loop
                print(f"[Hook Error] Event '{event}' failed: {exc}")

    async def execute_tool(self, block: ToolUseBlock) -> ToolResultBlockParam:
        """Run one tool call and build its tool_result block."""

        await self.emit("on_tool_call", call=block)
        tool_result = await self.tool_runtime.run_tool(
            block.name, cast(dict[str, Any], block.input)
        )
        await self.emit(
            "on_tool_result",
            result=tool_result,
            call_name=block.name,
            call_args=block.input,
        )
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": tool_result.result or tool_result.error or "",
        }

    async def step(self, messages: list[MessageParam]) -> ToolLoopResult:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=self.tool_runtime.tool_specs,
        ) as stream:
            async for text in stream.text_stream:
                await self.emit("on_text_delta", text=text)

            response = await stream.get_final_message()

        assistant_message: MessageParam = {
            "role": "assistant",
            "content": cast(Any, response.content),
        }

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        )

        await self.emit("on_model_response", text=text)

        # Check for and handle model tool use
        if response.stop_reason == "tool_use":
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_responses = list(
                await asyncio.gather(*(self.execute_tool(b) for b in tool_blocks))
            )

            return ToolLoopResult(
                assistant_message=assistant_message,
                text=text,
                tool_results=tool_responses,
            )

        return ToolLoopResult(assistant_message=assistant_message, text=text)

    async def loop(self, messages: list[MessageParam]) -> None:
        """Step until the model stops calling tools, appending to `messages`."""
        while True:
            tool_loop_result = await self.step(messages)
            assistant_message = tool_loop_result.assistant_message
            messages.append(assistant_message)
            if tool_loop_result.tool_results:
                messages.append(
                    {"role": "user", "content": tool_loop_result.tool_results}
                )
            else:
                break
