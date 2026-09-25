import asyncio

from anthropic.types import MessageParam
from dotenv import load_dotenv

from .agent import Agent
from .tools import AgentToolRuntime, Bash, Edit, ReadFile, Write

load_dotenv()


async def amain(agent: Agent, messages: list[MessageParam]) -> None:
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = (await asyncio.to_thread(input, "\nYou: ")).strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        await agent.loop(messages)


def main() -> None:
    messages: list[MessageParam] = []
    agent_tool_runtime = AgentToolRuntime(tools=[ReadFile, Write, Edit, Bash])
    agent = Agent(agent_tool_runtime)
    asyncio.run(amain(agent, messages))
