#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Dynamic batching probe.

The `items` parameter carries `BatchConfig(mode="dynamic", capacity=4)`,
so muna-server's `BatchPlan::from_signature` derives `Buffered` with 
`wait_full: false`: requests are merged up to capacity or until the 
flush deadline (100ms) elapses, whichever comes  first, then split back per caller.

The predictor echoes each item (prefixed with the broadcast `prefix`
param) into a dict stamped with the invocation's shared `start`/`end`
timestamps. Requests merged into the same batch return IDENTICAL
timestamps (proof of a single merged invocation); the echoed `item`
values prove each caller got its own slice back. `prefix` is a
NON-batch (broadcast) parameter, so it participates in the dispatcher's
batch key: requests with different `prefix` values must NOT merge - the
mismatched request is held for the next batch and lands in a separate
invocation with different timestamps.

Consumed by muna-server `tests/serving.rs`: the buffered merge/split
tests over `/v1/predictions/remote`, including the mismatched-key hold
and the partial-batch flush at the deadline.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, BatchConfig, Parameter
from time import sleep, time
from typing import Annotated

@compile(
    tag="@muna/test-batch-dynamic",
    access="unlisted"
)
def batch_dynamic(
    items: Annotated[list[str], Parameter.Generic(
        description="Items to echo, merged across concurrent requests.",
        batch=BatchConfig(mode="dynamic", capacity=4)
    )],
    *,
    prefix: Annotated[str, Parameter.Generic(
        description="""
        Broadcast prefix applied to every echoed item. 
        Participates in the dispatcher's batch key.
        """
    )]=""
) -> Annotated[
    list[dict],
    Parameter.Generic(description="Per-item echo with shared invocation timestamps.")
]:
    """
    Echo each prefixed item with the invocation's shared wall-clock timestamps.
    """
    start = time()
    sleep(0.2)
    end = time()
    return [
        { "item": f"{prefix}{item}", "start": start, "end": end }
        for item in items
    ]

if __name__ == "__main__":
    print(batch_dynamic(["a", "b"], prefix="x:"))
