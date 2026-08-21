"""Tests for the analytical properties of the abstract Y^{nN} DHT kernel.

These properties come directly from baddour2019.md and require no input signal
beyond the kernel itself (or an arbitrary vector, for the round-trip/Parseval
checks) -- they hold regardless of what the transformed data represents.
"""

import numpy as np
import pytest
from scipy.special import jv

from pypft.dht import _IMPLEMENTATIONS

from .conftest import ATOL, DHT_ORDERS, RTOL, SIGNAL_SIZE
from .tolerance import dht_tolerance


def test_kernel_is_self_inverse(implementation, order):
    """Y^{nN} Y^{nN} = I (baddour2019.md, Eq. 41), even though Y is not symmetric.

    This is the regression test for the removed ``RecurrenceBesselDHT``'s
    divergence: the residual grows with ``order``, so it is bounded by the
    ``dht_tolerance`` model rather than a flat tolerance -- a flat bound would
    either hide a regression like the deleted ``RecurrenceBesselDHT``'s (too
    loose) or reject the numerically-correct kernel above order ~24 (too
    tight).
    """
    kernel, _ = _IMPLEMENTATIONS[implementation]._bessel_kernel(order, SIGNAL_SIZE)
    identity = kernel @ kernel
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(identity, np.eye(SIGNAL_SIZE), rtol=tol, atol=tol)


@pytest.mark.parametrize("n", DHT_ORDERS, ids=lambda n: f"n={n}")
def test_kernel_self_inverse_small_n_looser_tolerance(implementation, n):
    """Eq. 41 holds even for small N, within baddour2019.md's ~1e-3 worst case."""
    size = 4
    kernel, _ = _IMPLEMENTATIONS[implementation]._bessel_kernel(n, size)
    identity = kernel @ kernel
    np.testing.assert_allclose(identity, np.eye(size), rtol=1e-2, atol=1e-3)


def test_kernel_application_is_self_inverse(implementation, order):
    """Applying the abstract kernel twice returns the original vector.

    Order-sensitive for the same reason as ``test_kernel_is_self_inverse``.
    """
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(SIGNAL_SIZE)
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    twice = impl._apply(kernel, impl._apply(kernel, vector))
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(twice, vector, rtol=tol, atol=tol)


def test_kronecker_delta_transform_pair(implementation, order):
    """The DHT of a standard basis vector is the matching kernel column (Eq. 65)."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    k0 = SIGNAL_SIZE // 3
    delta = np.zeros(SIGNAL_SIZE)
    delta[k0] = 1.0
    transformed = impl._apply(kernel, delta)
    np.testing.assert_allclose(transformed, kernel[:, k0], rtol=RTOL, atol=ATOL)


def test_weighted_parseval_is_preserved(implementation, order):
    """Y^{nN} preserves the J_{n+1}(j_nk)^{-2}-weighted inner product (Eq. 58-59).

    Unlike the (symmetric) T^{nN} formulation, Y^{nN} does not preserve the raw
    inner product <p, q> under transformation -- only this weighted one.
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, zeros = impl._bessel_kernel(order, SIGNAL_SIZE)
    weight = 1.0 / jv(order + 1, zeros[:-1]) ** 2

    rng = np.random.default_rng(1)
    p = rng.standard_normal(SIGNAL_SIZE)
    q = rng.standard_normal(SIGNAL_SIZE)
    transformed_p = impl._apply(kernel, p)
    transformed_q = impl._apply(kernel, q)

    lhs = np.sum(transformed_p * transformed_q * weight)
    rhs = np.sum(p * q * weight)
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(lhs, rhs, rtol=tol, atol=tol)


def test_forward_inverse_round_trip_is_exact_up_to_tolerance(implementation, order):
    """The physical (R-scaled) forward/inverse transform round-trips exactly.

    Order-sensitive for the same reason as ``test_kernel_is_self_inverse``.
    """
    rng = np.random.default_rng(2)
    f = rng.standard_normal(SIGNAL_SIZE)
    impl = _IMPLEMENTATIONS[implementation]
    R = 1.0
    F = impl.forward(f, order, R)
    reconstructed = impl.inverse(F, order, R)
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(reconstructed, f, rtol=tol, atol=tol)
