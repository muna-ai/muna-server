#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Continuous batching probe.

The `items` parameter carries `BatchConfig(mode="continuous")`, so muna-server's
`BatchPlan::from_signature` derives `Continuous`: the dispatcher submits
every request straight to the engine with NO lock and NO buffering,
because a continuously-batching engine owns synchronization internally.

The predictor sleeps ~400ms and echoes each item with `start`/`end`
timestamps. When the Rust test fires concurrent requests, their
`[start, end]` windows MUST overlap - if the dispatcher wrongly
serialized continuous mode (the bug this guards against), the windows
would be disjoint like the sequential probe's.

Consumed by muna-server `tests/serving.rs`: the continuous-overlap
test over `/v1/predictions/remote`.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, BatchConfig, Parameter
from time import sleep, time
from typing import Annotated

@compile(
    tag="@muna/test-batch-continuous",
    access="unlisted"
)
def batch_continuous(
    items: Annotated[list[str], Parameter.Generic(
        description="Items to echo. The engine is presumed to batch continuously.",
        batch=BatchConfig(mode="continuous")
    )]
) -> Annotated[
    list[dict],
    Parameter.Generic(description="Per-item echo with invocation timestamps.")
]:
    """
    Echo each item with the invocation's wall-clock timestamps.
    """
    start = time()
    sleep(0.4)
    end = time()
    return [
        { "item": item, "start": start, "end": end }
        for item in items
    ]

if __name__ == "__main__":
    print(batch_continuous(["a", "b"]))
