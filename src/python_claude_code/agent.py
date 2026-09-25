import asyncio
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam

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

            for block in tool_blocks:
                print(f"Tool Call: {block.name}:{block.input}")

            results = await asyncio.gather(
                *(
                    self.tool_runtime.run_tool(
                        block.name, cast(dict[str, Any], block.input)
                    )
                    for block in tool_blocks
                )
            )

            tool_responses: list[ToolResultBlockParam] = []
            for block, tool_result in zip(tool_blocks, results):
                tool_responses.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result.result or tool_result.error or "",
                    }
                )
                print(f"tool result: {tool_result}")

            return ToolLoopResult(
                assistant_message=assistant_message,
                text=text,
                tool_results=tool_responses,
            )

        return ToolLoopResult(assistant_message=assistant_message, text=text)
