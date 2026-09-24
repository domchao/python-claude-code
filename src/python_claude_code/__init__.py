import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

from .loop import run_loop
from .tools import AgentToolRuntime, Bash, Edit, ReadFile, Write

load_dotenv()

client = anthropic.Anthropic()
model = "claude-haiku-4-5-20251001"


messages: list[MessageParam] = []
agent_tool_runtime = AgentToolRuntime(tools=[ReadFile, Write, Edit, Bash])


def main() -> None:
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        while True:
            tool_loop_result = run_loop(
                client=client,
                model=model,
                messages=messages,
                tool_runtime=agent_tool_runtime,
            )
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
