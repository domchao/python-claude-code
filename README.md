# Python 'Claude Code'

A simple AI coding agent written in Python. Read / Write / Edit / Bash

## Installation

```bash
uv sync
```

## Usage

Run the CLI with:

```bash
python-claude-code
```

This starts an interactive session where you can give instructions to the AI agent. For example:

```
Type 'exit' or 'quit' to stop.

You: Create a new file called hello.txt with "Hello, World!"
Assistant: I'll create that file for you.
[File operations...]

You: Read the hello.txt file
Assistant: The file contains "Hello, World!"

You: exit
```

The agent can read, write, edit files, and execute bash commands based on your instructions.
