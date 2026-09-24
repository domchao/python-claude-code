import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv

from .loop import run_loop

load_dotenv()

client = anthropic.Anthropic()
model = "claude-haiku-4-5-20251001"

USER_INPUT = "Tell me a joke about ducks"

messages: list[MessageParam] = [
    {"role": "user", "content": USER_INPUT},
]


def main() -> None:
    message = run_loop(client=client, model=model, messages=messages)
    print(message)
