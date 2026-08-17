#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Cold-start probe.

Every prediction sleeps ~12 seconds, including the
warmup prediction muna-server's registry issues when a model is first
loaded. That holds the registry's `Loading` sentinel PAST the registry's
10-second hold threshold (requests wait on a loading model up to
`HOLD_THRESHOLD` before giving up), so the Rust test can deterministically
observe the loading window: requests that arrive mid-load must be
rejected with `503` and a `Retry-After` header, and `/status` must report
the model as `loading` before it flips to `ready`.

Consumed by muna-server `tests/serving.rs`: the loading-window test
(503 + Retry-After during load, then eventual readiness).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["muna"]
# ///

from muna import compile, Parameter
from time import sleep, time
from typing import Annotated

@compile(
    tag="@muna/test-slow-coldstart",
    access="unlisted"
)
def slow_coldstart(
    value: Annotated[
        str,
        Parameter.Generic(description="Value to echo back.")
    ]
) -> Annotated[
    str,
    Parameter.Generic(description="Echoed value, after a ~12s delay.")
]:
    """
    Echo the input after a deliberate ~12 second delay.
    """
    sleep(12.)
    return value

if __name__ == "__main__":
    start = time()
    print(slow_coldstart("hello"), f"({time() - start:.1f}s)")
