#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Sequential-dispatch probe.

No parameter carries a `BatchConfig`, so muna-server's 
`BatchPlan::from_signature` derives `Sequential` and guards
every prediction with a per-model mutex.

The predictor sleeps ~400ms and returns its input alongside `start`/`end`
wall-clock timestamps. When the Rust test fires two concurrent requests,
serialized dispatch means their `[start, end]` windows must NOT overlap;
if the guard were broken, the sleeps would run concurrently and the
windows would interleave.

Consumed by muna-server `tests/serving.rs`: the sequential-guard test
over `/v1/predictions/remote`.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, Parameter
from time import sleep, time
from typing import Annotated

@compile(
    tag="@muna/test-batch-sequential",
    access="unlisted"
)
def batch_sequential(
    value: Annotated[
        str,
        Parameter.Generic(description="Value to echo back.")
    ]
) -> Annotated[
    dict,
    Parameter.Generic(description="Echoed value with dispatch timestamps.")
]:
    """
    Echo the input along with wall-clock dispatch timestamps.
    """
    start = time()
    sleep(0.4)
    return { "value": value, "start": start, "end": time() }

if __name__ == "__main__":
    print(batch_sequential("hello"))
