from anthropic import Client
from anthropic.types import MessageParam


def run_loop(client: Client, model: str, messages: list[MessageParam]):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )

    for block in message.content:
        if block.type == "text":
            print(block.text)

    return message
