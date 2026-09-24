#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
OpenAI-compatible fake embedding model.

Embeddings are hash-derived: each text seeds a NumPy `RandomState` with the
first four bytes of its SHA-256 digest, so the same text always produces the
same vector (across runs and processes) without any model. The Usage output
reports one token per whitespace-separated word.

Consumed by muna-server `tests/serving.rs`: `/v1/embeddings` shape
(`[n, dims]` float vectors), determinism (identical vectors across calls),
and usage propagation.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from hashlib import sha256
from muna import compile, Parameter
from muna.beta import Annotations
from numpy import float32, ndarray, stack
from numpy.random import RandomState
from pydantic import BaseModel
from typing import Annotated

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

@compile(
    tag="@muna/test-openai-embeddings",
    access="unlisted"
)
def embedding_model(
    texts: Annotated[
        list[str],
        Parameter.Generic(description="Input texts to embed.")
    ],
    *,
    dimensions: Annotated[int, Annotations.EmbeddingDims(
        description="Embedding dimensions.",
        min=32,
        max=1024
    )]=1024
) -> tuple[
    Annotated[
        ndarray,
        Parameter.Embedding(description="Embedding matrix.")
    ],
    Annotated[
        Usage,
        Parameter.Generic(description="Token usage.")
    ]
]:
    """
    Fake model compatible with the OpenAI Embeddings API.
    """
    embeddings = stack([_embed(text, dimensions) for text in texts])
    tokens = sum(len(text.split()) for text in texts)
    return embeddings, Usage(prompt_tokens=tokens, total_tokens=tokens)

def _embed(text: str, dimensions: int) -> ndarray:
    seed = int.from_bytes(sha256(text.encode()).digest()[:4], "little")
    rng = RandomState(seed)
    return rng.standard_normal(dimensions).astype(float32)

if __name__ == "__main__":
    embeddings, usage = embedding_model(
        [
            "What is the capital of France?",
            "Butterflies have legs."
        ],
        dimensions=64
    )
    print(embeddings.shape, embeddings.dtype)
    print(usage)
