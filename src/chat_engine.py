"""Chat engine: drives a Claude-powered conversation that can call scikit-learn tools.

Uses a manual tool-use loop against the Messages API (see the Anthropic API docs)
rather than the beta tool runner, since Streamlit re-executes the whole script on
every interaction and a plain request/response loop is easiest to reason about here.
"""
from __future__ import annotations

import anthropic

from src.chat_tools import TOOLS, make_executor

MODEL = "claude-opus-5"

SYSTEM_PROMPT = (
    "You are a data science assistant embedded in a Streamlit app that wraps scikit-learn. "
    "You can list datasets, describe them, list available algorithms, and train/evaluate "
    "models via tools. Always use the tools rather than guessing results or metrics. When "
    "you report metrics, format them clearly and briefly explain what they mean. If a tool "
    "call fails, explain the error to the user and suggest a fix (e.g. a valid target column "
    "or dataset name)."
)


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def run_turn(messages: list, workspace: dict) -> list:
    """Run one user turn to completion (including any tool calls).

    `messages` must already end with the new user message. Returns the updated
    list, ending with the assistant's final (non-tool-use) response.
    """
    client = get_client()
    execute = make_executor(workspace)

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = execute(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            except Exception as exc:  # tool errors are reported back to Claude, not raised
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(exc),
                        "is_error": True,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return messages
