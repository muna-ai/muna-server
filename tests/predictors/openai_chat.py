#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
OpenAI-compatible fake chat model.

The generator emits a fixed `reasoning_content` delta (DeepSeek reasoning
convention), then deterministically echoes the last user message
word-by-word, one chunk per word, then emits a final usage-bearing chunk
whose `prompt_tokens_details.cached_tokens` equals the number of user
messages in the conversation and whose
`completion_tokens_details.reasoning_tokens` counts the reasoning words.

A last user message of exactly `__invalid__` raises `ValueError` before the
first yield, standing in for a caller fault such as a prompt longer than the
model's context length. Servers must answer it with HTTP 400 for both
streaming and non-streaming requests.

Consumed by `tests/serving.rs`: non-streaming chat shape, SSE streaming
reassembly, `usage.prompt_tokens_details.cached_tokens` plumbing,
`reasoning_content` passthrough (streamed deltas and accumulated message),
and pre-output caller faults rendering as 400.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from json import dumps
from muna import compile, Parameter
from muna.beta import Annotations
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk,
    DeltaMessage, Message, StreamChoice
)
from time import sleep, time
from typing import Annotated, Iterator

@compile(
    tag="@muna/test-openai-chat",
    access="unlisted"
)
def chat_model(
    messages: Annotated[
        list[Message],
        Parameter.Generic(description="Messages comprising the chat conversation so far.")
    ],
    *,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.,
        max=2.
    )]=1.,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=4_096
    )]=512
) -> Iterator[ChatCompletionChunk]:
    """
    Fake model compatible with the OpenAI Chat Completions API.
    """
    completion_id = "chatcmpl-test"
    created = int(time())
    user_messages = [
        message
        for message in messages
        if message["role"] == "user"
    ]
    last_user_content = _text(user_messages[-1]["content"]) if user_messages else ""
    # Caller-fault trigger. Raised before the first yield, as real predictors
    # do for context-length overflow, so `muna-rs` classifies it as
    # `InvalidInput` and the server answers 400 instead of committing a 200
    # stream that carries an error frame.
    if last_user_content == "__invalid__":
        raise ValueError("The input is longer than the model's context length.")
    words = last_user_content.split()[:max_output_tokens]
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="test-openai-chat",
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content="")
            )
        ]
    )
    # Reasoning delta precedes content, mirroring how real reasoning models
    # stream. A delta never carries both `reasoning_content` and `content`.
    reasoning = "thinking really hard"
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="test-openai-chat",
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(reasoning_content=reasoning)
            )
        ]
    )
    for idx, word in enumerate(words):
        sleep(0.5) # pretend like we are doing useful work
        content = word if idx == 0 else f" {word}"
        yield ChatCompletionChunk(
            id=completion_id,
            created=created,
            model="test-openai-chat",
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=content)
                )
            ]
        )
    prompt_tokens = sum([len(_text(message["content"]).split()) for message in messages])
    yield ChatCompletionChunk(
        id=completion_id,
        created=created,
        model="test-openai-chat",
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletion.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(words),
            total_tokens=prompt_tokens + len(words),
            prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                cached_tokens=len(user_messages)
            ),
            completion_tokens_details=ChatCompletion.Usage.CompletionTokensDetails(
                reasoning_tokens=len(reasoning.split())
            )
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
    from rich import print, print_json
    stream = chat_model([
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="the quick brown fox")
    ])
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(f"[green]{delta.content}[/green]", end="")
        if chunk.usage:
            print("\nusage: ", end="")
            print_json(chunk.usage.model_dump_json())