import asyncio

from anthropic.types import MessageParam
from dotenv import load_dotenv

from .agent import Agent
from .tools import AgentToolRuntime, Bash, Edit, ReadFile, Write

load_dotenv()

messages: list[MessageParam] = []
agent_tool_runtime = AgentToolRuntime(tools=[ReadFile, Write, Edit, Bash])
agent = Agent(agent_tool_runtime)


async def amain() -> None:
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = (await asyncio.to_thread(input, "\nYou: ")).strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            tool_loop_result = await agent.step(messages)
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


def main() -> None:
    asyncio.run(amain())
