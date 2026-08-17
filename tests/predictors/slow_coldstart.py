#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

"""
Cold-start probe.

The predictor is deliberately parameterless: muna-server's registry warms a
model with a sentinel prediction that omits all required inputs, and with no
required inputs that warmup prediction runs the full function body -- the
~12 second sleep therefore executes DURING load. That holds the registry's
`Loading` sentinel past its 10-second hold threshold (requests wait on a
loading model up to `HOLD_THRESHOLD` before giving up), so the Rust test can
deterministically observe the loading window: requests that arrive mid-load
must be rejected with `503` and a `Retry-After` header, and `/status` must
report the model as `loading` before it flips to `ready`.

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
def slow_coldstart() -> Annotated[
    str,
    Parameter.Generic(description="Constant marker, after a ~12s delay.")
]:
    """
    Sleep ~12 seconds, then return a constant marker.
    """
    sleep(12.)
    return "warm"

if __name__ == "__main__":
    start = time()
    print(slow_coldstart(), f"({time() - start:.1f}s)")
