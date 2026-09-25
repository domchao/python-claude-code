from dataclasses import dataclass
from typing import Any, cast

from anthropic import Client
from anthropic.types import MessageParam, ToolResultBlockParam

from .tools import AgentToolRuntime


@dataclass
class ToolLoopResult:
    assistant_message: MessageParam
    text: str = ""
    tool_results: list[ToolResultBlockParam] | None = None


def run_loop(
    client: Client,
    model: str,
    messages: list[MessageParam],
    tool_runtime: AgentToolRuntime,
) -> ToolLoopResult:
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=messages,
        tools=tool_runtime.tool_specs,
    )

    assistant_message: MessageParam = {
        "role": "assistant",
        "content": cast(Any, response.content),
    }

    text = "\n".join(block.text for block in response.content if block.type == "text")

    # Check for and handle model tool use
    if response.stop_reason == "tool_use":
        tool_responses: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"Tool Call: {block.name}:{block.input}")
            tool_result = tool_runtime.run_tool(
                block.name, cast(dict[str, Any], block.input)
            )
            tool_responses.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result.result or tool_result.error or "",
                }
            )
            print(f"tool result: {tool_result}")

        return ToolLoopResult(
            assistant_message=assistant_message, text=text, tool_results=tool_responses
        )

    return ToolLoopResult(assistant_message=assistant_message, text=text)
