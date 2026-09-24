#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
OpenAI-compatible fake tool-calling chat model.

With `tools` provided, the generator emits a scripted `get_weather` tool
call split across three fragments (id + name first, then two argument
fragments) ending with `finish_reason: "tool_calls"`. Without tools, it
echoes the last user message and finishes with `stop`.

Consumed by `tests/serving.rs`: tool-call fragment streaming, chunk-merge
accumulation into `message.tool_calls`, `finish_reason` passthrough, and
the Anthropic adapter's `tool_use` block + `input_json_delta` translation.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, Parameter
from muna.beta import Annotations
from json import dumps
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, ChoiceDeltaToolCall,
    DeltaMessage, Message, StreamChoice
)
from time import time
from typing import Annotated, Iterator

@compile(
    tag="@muna/test-openai-tools",
    access="unlisted"
)
def tools_chat(
    messages: Annotated[
        list[Message],
        Parameter.Generic(description="Messages comprising the chat conversation so far.")
    ],
    *,
    tools: Annotated[
        list[dict],
        Annotations.ChatTools(description="Tools the model may call.")
    ]=None
) -> Iterator[ChatCompletionChunk]:
    """
    Fake tool-calling model compatible with the OpenAI Chat Completions API.
    """
    completion_id = "chatcmpl-tools-test"
    created = int(time())
    tool_count = len(tools) if tools is not None else 0
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="test-openai-tools",
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content="")
        )]
    )
    if tool_count > 0:
        # Scripted tool call: id + name arrive on the first fragment, the
        # JSON-encoded arguments split across the next two.
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="test-openai-tools",
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(tool_calls=[ChoiceDeltaToolCall(
                    index=0,
                    id="call_test_0",
                    type="function",
                    function=ChoiceDeltaToolCall.Function(name="get_weather", arguments="")
                )])
            )]
        )
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="test-openai-tools",
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(tool_calls=[ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCall.Function(arguments="{\"location\": ")
                )])
            )]
        )
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="test-openai-tools",
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(tool_calls=[ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCall.Function(arguments="\"Paris\"}")
                )])
            )]
        )
        finish_reason = "tool_calls"
        completion_tokens = 3
    else:
        user_messages = [
            message
            for message in messages
            if message["role"] == "user"
        ]
        last_user_content = _text(user_messages[-1]["content"]) if user_messages else ""
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="test-openai-tools",
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(content=last_user_content)
            )]
        )
        finish_reason = "stop"
        completion_tokens = 1
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="test-openai-tools",
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason=finish_reason
        )],
        usage=ChatCompletion.Usage(
            prompt_tokens=len(messages),
            completion_tokens=completion_tokens,
            total_tokens=len(messages) + completion_tokens
        )
    )

def _text(content) -> str:
    """
    Flatten OpenAI message content (a plain string or a list of text parts)
    to a string. Compiled chat models get this from the tokenizer's chat
    template; this tokenizer-free fake reproduces it so echoes hold for
    both wire shapes.
    """
    if dumps(content).startswith("["):
        return "\n".join([part["text"] for part in content if part["type"] == "text"])
    return content

if __name__ == "__main__":
    from rich import print_json
    stream = tools_chat(
        [Message(role="user", content="What is the weather in Paris?")],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }
            }
        }]
    )
    for chunk in stream:
        print_json(chunk.model_dump_json(exclude_none=True))
