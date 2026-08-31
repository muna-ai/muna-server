#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Exercises the KV routing sidecar rail. Declaring
`KVRoutingMetadata(tokenize=...)` makes the compiler emit two sidecar
variants:

- `{tag}:router`: Runs `tokenize` (whose params must be a name-subset 
  of the predictor's) then paginates the token IDs into the SAME chained 
  SHA-256 page hashes the engine keys its KV cache with, returned as 
  64-hex strings. Token IDs never leave the sidecar.

- `{tag}:kv`: Returns the ZMQ endpoint the engine publishes KV cache
  events on.

Serving: muna-server relays `{tag}:kv` events to the control plane
(page hash -> node index); the plane runs `{tag}:router` on request
bodies and routes to the node with the longest cached prefix.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2", "muna", "transformers>=5.11"]
# ///

from muna import compile, Sandbox
from muna.beta import KVRoutingMetadata
from muna.beta.openai import Message
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nvidia/GLM-5.2-NVFP4")

def _tokenize(messages) -> list[int]:
    return tokenizer.apply_chat_template(
        [{ "role": m.role, "content": m.content } for m in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )

@compile(
    tag="@muna/test-kv-routing",
    access="unlisted",
    sandbox=Sandbox().pip_install("jinja2", "transformers>=5.11"),
    metadata=[
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def kv_routing_test(messages: list[Message]) -> str:
    """
    Predictor used to test KV routing sidecars.
    """
    tokens = _tokenize(messages)
    total_tokens = len(tokens)
    return f"There are {total_tokens} tokens in your prompt"

if __name__ == "__main__":
    result = kv_routing_test([
        Message(role="user", content="What is the capital of France?")
    ])
    print(result)