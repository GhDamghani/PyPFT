"""Tests for the ``dht_tolerance`` model itself (see ``tests.dht.tolerance``).

The model must stay an upper bound on the measured self-inverse residual --
otherwise a real regression (like the deleted ``RecurrenceBesselDHT``'s
divergence) could slip past the assertions that rely on it -- but it must not
be so loose that it stops meaning anything. Both directions are checked here.
"""

import numpy as np
import pytest

from pypft.dht._cached import CachedBesselDHT

from .conftest import DHT_ORDERS, SIGNAL_SIZE
from .tolerance import dht_tolerance

#: The model is allowed to overestimate the measured residual by at most this
#: factor (develop_plan.md §3.2's "~10x" margin), so it stays tight enough to
#: still catch a future regression rather than silently absorbing it.
MAX_MODEL_TO_MEASURED_RATIO = 10.0


@pytest.mark.parametrize("order", DHT_ORDERS, ids=lambda n: f"n={n}")
def test_dht_tolerance_bounds_measured_residual(order):
    """The model both bounds and stays within ~10x of the measured residual."""
    kernel, _ = CachedBesselDHT._bessel_kernel(order, SIGNAL_SIZE)
    residual = np.max(np.abs(kernel @ kernel - np.eye(SIGNAL_SIZE)))
    tol = dht_tolerance(order, SIGNAL_SIZE)

    assert residual <= tol, (
        f"order={order}: measured residual {residual:.3e} exceeds the model's "
        f"tolerance {tol:.3e} -- the model no longer bounds reality"
    )
    assert tol <= MAX_MODEL_TO_MEASURED_RATIO * residual, (
        f"order={order}: model tolerance {tol:.3e} is more than "
        f"{MAX_MODEL_TO_MEASURED_RATIO}x the measured residual {residual:.3e} "
        "-- the model is too loose to catch a future regression"
    )
