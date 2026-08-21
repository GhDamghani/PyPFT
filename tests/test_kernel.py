"""Tests for the explicit, brute-force PFT kernel oracle (``pypft._kernel``).

``kernel_matrix`` is built completely independently of ``forward_pft``/
``inverse_pft``'s own FFT-plus-per-harmonic-Hankel composition -- straight
from Yao & Baddour's own combined-operator definition (PeerJ CS Part II,
Eqs. 1-3) -- so applying it and comparing to the composed pipeline is a
stronger check than the Gaussian oracle (``tests/test_transform.py``), which
only certifies *accuracy* against a known analytic answer, not that the
fast composition and the direct definition agree exactly.
"""

import numpy as np
import pytest

from pypft._kernel import kernel_matrix
from pypft.grid import PolarGrid
from pypft.transform import Direction, forward_pft, inverse_pft

#: Small enough that building the O(n_radial**2 * n_angular**3) matrix stays
#: fast, while still exercising several distinct harmonic orders.
_GRID_CASES = [
    (8, 5, 3.0),
    (10, 6, 5.0),  # even n_angular (Section 3.4's parity caveat)
]


@pytest.mark.parametrize("n_radial,n_angular,R", _GRID_CASES)
def test_forward_kernel_matrix_reproduces_forward_pft(n_radial, n_angular, R):
    """``kernel_matrix(..., FORWARD) @ f.ravel()`` matches ``forward_pft(f)``."""
    grid = PolarGrid(n_radial=n_radial, n_angular=n_angular, R=R)
    rng = np.random.default_rng(0)
    f = rng.standard_normal((n_radial, n_angular))

    expected = forward_pft(f=f, grid=grid)
    operator = kernel_matrix(grid=grid, direction=Direction.FORWARD)
    actual = (operator @ f.ravel()).reshape(n_radial, n_angular)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("n_radial,n_angular,R", _GRID_CASES)
def test_inverse_kernel_matrix_reproduces_inverse_pft(n_radial, n_angular, R):
    """``kernel_matrix(..., INVERSE) @ F.ravel()`` matches ``inverse_pft(F)``."""
    grid = PolarGrid(n_radial=n_radial, n_angular=n_angular, R=R)
    rng = np.random.default_rng(1)
    F = rng.standard_normal((n_radial, n_angular)) + 1j * rng.standard_normal(
        (n_radial, n_angular)
    )

    expected = inverse_pft(F=F, grid=grid)
    operator = kernel_matrix(grid=grid, direction=Direction.INVERSE)
    actual = (operator @ F.ravel()).reshape(n_radial, n_angular)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_kernel_matrix_rejects_a_non_direction_member():
    """``kernel_matrix`` type-validates its ``direction`` argument."""
    grid = PolarGrid(n_radial=8, n_angular=5, R=3.0)
    bad_direction: Direction = "forward"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        kernel_matrix(grid=grid, direction=bad_direction)
