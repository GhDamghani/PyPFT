"""The angular DFT of a pure complex harmonic is a single centered delta.

``x[p] = exp(1j * n0 * theta_p)``, sampled at the centered angle grid
``theta_p = -pi + b/2 + p*b`` (``b = 2*pi/n_angular``), is exactly the
harmonic-``n0`` basis function of the angular DFT. Its transform must
therefore be zero everywhere except at the centered index predicted by
``pypft.dft.harmonics`` -- this is the sharpest possible check that
``angular_dft`` centers its output the way ``pypft.axes`` says every stored
array is centered, for both angular parities and at both edges of the
harmonic range (a single ``n0`` cannot see a sign flip in the centering).
"""

import numpy as np
import pytest

from pypft.dft import angular_dft, harmonics

#: ``(n_angular, n0)`` pairs spanning every harmonic of both a odd and an
#: even angular sample count, including both edges of each harmonic range.
_CASES = [
    (n_angular, int(n0))
    for n_angular in (15, 16)
    for n0 in harmonics(n_angular=n_angular)
]


@pytest.mark.parametrize("n_angular,n0", _CASES, ids=lambda v: str(v))
def test_forward_dft_of_a_pure_harmonic_is_a_single_centered_delta(
    n_angular, n0, implementation
):
    """``angular_dft`` of ``exp(1j*n0*theta)`` is a delta at ``n0 + n_angular//2``."""
    b = 2 * np.pi / n_angular
    p = np.arange(n_angular)
    theta = -np.pi + b / 2 + p * b
    x = np.exp(1j * n0 * theta)

    result = angular_dft(x=x, implementation=implementation)

    target_index = n0 + n_angular // 2
    expected = np.zeros(n_angular, dtype=complex)
    # x[n_angular // 2] is exactly the phase the natural-order FFT sees at its
    # own q=0 after un-centering (ifftshift moves that centered sample there),
    # so it is the delta's own residual phase -- no separate formula needed.
    expected[target_index] = n_angular * x[n_angular // 2]

    np.testing.assert_allclose(result, expected, atol=1e-9)
