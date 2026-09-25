import asyncio
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock

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

    async def execute_tool(self, block: ToolUseBlock) -> ToolResultBlockParam:
        """Run one tool call and build its tool_result block."""
        print(f"Tool Call: {block.name}:{block.input}")
        tool_result = await self.tool_runtime.run_tool(
            block.name, cast(dict[str, Any], block.input)
        )
        print(f"tool result: {tool_result}")
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": tool_result.result or tool_result.error or "",
        }

    async def step(self, messages: list[MessageParam]) -> ToolLoopResult:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=self.tool_runtime.tool_specs,
        )

        assistant_message: MessageParam = {
            "role": "assistant",
            "content": cast(Any, response.content),
        }

        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        )

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
                if tool_loop_result.text:
                    print(f"\nAssistant: {tool_loop_result.text}")
                break
