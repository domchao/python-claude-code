import asyncio

from anthropic.types import MessageParam
from dotenv import load_dotenv

from .agent import Agent
from .hooks import print_llm_response, print_tool_call, print_tool_result
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
    # register hooks
    agent.on("on_model_response", print_llm_response)
    agent.on("on_tool_call", print_tool_call)
    agent.on("on_tool_result", print_tool_result)

    asyncio.run(amain(agent, messages))
