"""Chat engine: drives a GPT-powered conversation that can call scikit-learn tools.

Uses a manual tool-call loop against the OpenAI Chat Completions API, since
Streamlit re-executes the whole script on every interaction and a plain
request/response loop is easiest to reason about here.

The OpenAI API key is supplied per call by the caller (typed into the Chat
Assistant page's sidebar and kept in that browser session's Streamlit state)
rather than read from an environment variable, so no key is ever hardcoded
or shared across sessions.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from src.chat_tools import TOOLS, make_executor

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

SYSTEM_PROMPT = (
    "You are a data science assistant embedded in a Streamlit app that wraps scikit-learn. "
    "You can list datasets, describe them, list available algorithms, and train/evaluate "
    "models via tools. Always use the tools rather than guessing results or metrics. When "
    "you report metrics, format them clearly and briefly explain what they mean. If a tool "
    "call fails, explain the error to the user and suggest a fix (e.g. a valid target column "
    "or dataset name)."
)


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def new_conversation() -> list:
    """A fresh message history, seeded with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_turn(messages: list, workspace: dict, api_key: str) -> list:
    """Run one user turn to completion (including any tool calls).

    `messages` must already end with the new user message. `api_key` is the
    OpenAI API key entered by the user for this session — it is never read
    from an environment variable or persisted anywhere. Returns the updated
    list, ending with the assistant's final (non-tool-call) response.
    """
    client = get_client(api_key)
    execute = make_executor(workspace)

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message

        assistant_entry = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        messages.append(assistant_entry)

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
                result = execute(tool_call.function.name, args)
            except Exception as exc:  # tool errors are reported back to the model, not raised
                result = f"Error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return messages
