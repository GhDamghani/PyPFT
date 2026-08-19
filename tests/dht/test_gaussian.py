"""Analytical validation against a known continuous Hankel transform pair.

The zeroth-order continuous Hankel transform of ``f(r) = exp(-r^2 / 2)`` is
``F(rho) = exp(-rho^2 / 2)`` -- an exact self-reciprocal pair. ``R = 8`` is chosen
so the Gaussian's tail is negligible at the space limit (``exp(-32) ~ 1e-14``)
and, given ``SIGNAL_SIZE = 64`` samples, the induced band limit ``W_rho = j_0N/R``
comfortably exceeds the Gaussian's own significant frequency support -- i.e. the
function is "effectively" both space- and band-limited, per baddour2019.md Sec. 4.
"""

import numpy as np

from pypft.dht import hankel_transform, sample_points

from .conftest import SIGNAL_SIZE

_ORDER = 0
_R = 8.0


def test_gaussian_matches_continuous_hankel_transform(implementation):
    """The DHT of a self-reciprocal Gaussian matches its known continuous transform."""
    r, rho = sample_points(_ORDER, SIGNAL_SIZE, _R, implementation)
    f = np.exp(-(r**2) / 2)
    F = hankel_transform(f, _ORDER, _R, implementation)
    F_expected = np.exp(-(rho**2) / 2)
    np.testing.assert_allclose(F, F_expected, rtol=1e-6, atol=1e-8)
