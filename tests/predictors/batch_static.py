#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Static batching probe.

The `items` parameter carries `BatchConfig(mode="static", capacity=4)`,
so muna-server's `BatchPlan::from_signature` derives
`Buffered { capacity: 4 }`: compatible concurrent requests are merged
into one invocation and the list output is split back per caller by
item count. Static and dynamic dispatch identically on the server --
the server never pads a partial batch; a predictor compiled with a
rigid batch shape must pad internally.

The predictor echoes each item into a dict stamped with the invocation's
shared `start`/`end` timestamps. Requests merged into the same batch
therefore return IDENTICAL timestamps - that is how the Rust test proves
a single merged invocation served multiple callers - while the echoed
`item` values prove the split assigned each caller its own slice.

Consumed by muna-server `tests/serving.rs`: the buffered merge/split
tests over `/v1/predictions/remote` (same-key requests merge; a
mismatched-key request is held for the next batch).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, BatchConfig, Parameter
from time import sleep, time
from typing import Annotated

@compile(
    tag="@muna/test-batch-static",
    access="unlisted"
)
def batch_static(
    items: Annotated[list[str], Parameter.Generic(
        description="Items to echo, merged across concurrent requests.",
        batch=BatchConfig(mode="static", capacity=4)
    )]
) -> Annotated[
    list[dict],
    Parameter.Generic(description="Per-item echo with shared invocation timestamps.")
]:
    """
    Echo each item with the invocation's shared wall-clock timestamps.
    """
    start = time()
    sleep(0.2)
    end = time()
    return [
        { "item": item, "start": start, "end": end }
        for item in items
    ]

if __name__ == "__main__":
    print(batch_static(["a", "b"]))
