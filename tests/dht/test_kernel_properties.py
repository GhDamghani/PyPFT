"""Tests for the analytical properties of the abstract Y^{nN} DHT kernel.

These properties come directly from baddour2019.md and require no input signal
beyond the kernel itself (or an arbitrary vector, for the round-trip/Parseval
checks) -- they hold regardless of what the transformed data represents.
"""

import numpy as np
import pytest
from scipy.special import jn_zeros, jv

from pypft.dht import _IMPLEMENTATIONS

from .conftest import ATOL, DHT_ORDERS, RTOL, SIGNAL_SIZE
from .tolerance import dht_tolerance

# ==========================================================================================
# Shared helpers for the shift/modulation/multiplication/convolution rule tests
# ==========================================================================================


def _general_bessel_kernel(n: int, size: int) -> np.ndarray:
    """Build ``Y^{nN}`` directly from ``scipy.special``, for any integer ``n``.

    Unlike ``BaseDHT._bessel_kernel`` (whose public entry points enforce
    ``n >= 0``, per ``pypft.transform``'s negative-order sign relation), this
    accepts a negative ``n`` too: the zeros of ``J_{-n}`` coincide with those
    of ``J_n`` because ``J_{-n}(x) = (-1)^n * J_n(x)`` is an exact identity
    for every ``x``, not only at a zero, so ``jn_zeros(abs(n), ...)`` is the
    right zero set for either sign of ``n``.

    :param n: The (possibly negative) order of the discrete Hankel transform.
    :type n: int
    :param size: The length of the vectors the kernel transforms.
    :type size: int
    :returns: The ``Y^{nN}`` kernel matrix (baddour2019.md, Eq. 39).
    :rtype: np.ndarray

    """
    zeros = jn_zeros(abs(n), size + 1)
    j_nN = zeros[-1]
    jn_vals = zeros[:-1]
    j_np1_vals = jv(n + 1, jn_vals)
    outer = np.outer(jn_vals, jn_vals) / j_nN
    kernel = (2.0 / j_nN) * jv(n, outer) / j_np1_vals[np.newaxis, :] ** 2
    return kernel


def _generalized_shift(
    kernel: np.ndarray, transform: np.ndarray, k0: int
) -> np.ndarray:
    """Compute the DHT's generalized shift of a vector, by ``k0``, in one domain.

    ``f^{shift}_{k,k0} = sum_p Y[k,p] * (Y[p,k0] * F[p])`` (baddour2019.md,
    Eq. 70): the shifted *space*-domain vector, built from ``F`` -- the
    *transform* of the vector being shifted -- since the classical
    ``f_{k-k0}`` shift is unusable here (``k - k0`` can fall outside
    ``[1, N-1]`` and the Bessel kernel, unlike the DFT's exponential, is not
    periodic). Self-inverse ``Y`` also makes this formula direction-agnostic:
    passing a frequency-domain vector shifts it in the space domain, and
    vice versa, which is what lets one helper serve both the shift-modulation
    and modulation-shift rules below.

    :param kernel: The ``(size, size)`` kernel matrix from ``_bessel_kernel``.
    :type kernel: np.ndarray
    :param transform: The transform of the vector being shifted.
    :type transform: np.ndarray
    :param k0: The shift amount, an index into ``transform``.
    :type k0: int
    :returns: The generalized-shifted vector.
    :rtype: np.ndarray

    """
    return kernel @ (kernel[:, k0] * transform)


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


@pytest.mark.parametrize("n", DHT_ORDERS, ids=lambda n: f"n={n}")
def test_negative_order_kernel_relates_by_sign(n):
    """``Y^{(-n)N} = (-1)^n * Y^{nN}`` exactly (``pypft.transform``'s own sign relation).

    Neither baddour2019.md nor the Part I paper states this identity for a
    negative integer order -- it is derived independently from
    ``J_{-n}(x) = (-1)^n * J_n(x)`` (an exact identity for every ``x``, not
    only at a zero of ``J_n``), which is why ``_general_bessel_kernel`` is
    used here instead of the package's own ``n >= 0``-only kernel builder.
    """
    positive = _general_bessel_kernel(n, SIGNAL_SIZE)
    negative = _general_bessel_kernel(-n, SIGNAL_SIZE)
    expected = ((-1.0) ** n) * positive
    np.testing.assert_allclose(negative, expected, rtol=RTOL, atol=ATOL)


def test_dht_shift_modulation_rule(implementation, order):
    """The DHT of a generalized-shifted vector is a plain entrywise product (Eq. 76-77).

    ``DHT(shift(F, k0))[m] == Y[m, k0] * F[m]`` -- the discrete analogue of
    the classical continuous rule ``FT{f(x-a)} = exp(-i*a*w) * f_hat(w)``.
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    rng = np.random.default_rng(3)
    g = rng.standard_normal(SIGNAL_SIZE)
    G = impl._apply(kernel, g)
    k0 = SIGNAL_SIZE // 4

    shifted = _generalized_shift(kernel, G, k0)
    lhs = impl._apply(kernel, shifted)
    rhs = kernel[:, k0] * G
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(lhs, rhs, rtol=tol, atol=tol)


def test_dht_modulation_rule(implementation, order):
    """Modulating by a kernel column in space is a generalized shift in frequency (Eq. 78-82).

    ``f = Y[:, k0] * g  =>  DHT(f) == shift(g, k0)`` -- the dual of the
    shift-modulation rule above, using the same ``_generalized_shift``
    formula (direction-agnostic because ``Y`` is self-inverse).
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    rng = np.random.default_rng(4)
    g = rng.standard_normal(SIGNAL_SIZE)
    k0 = SIGNAL_SIZE // 4

    modulated = kernel[:, k0] * g
    lhs = impl._apply(kernel, modulated)
    rhs = _generalized_shift(kernel, g, k0)
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(lhs, rhs, rtol=tol, atol=tol)


def test_dht_convolution_rule(implementation, order):
    """The DHT of a generalized convolution is a plain (Hadamard) product (Eq. 83-86).

    ``(g*h)_k := sum_k0 g[k0] * shift(H, k0)[k]``, and this convolution's own
    DHT equals ``G * H`` entrywise -- resolving what baddour2019.md notes the
    *classical* (continuous) Hankel transform lacks in general, since
    ``J_n(x-y)`` has no simple relationship to ``J_n(x)``. Also checks the
    convolution is commutative (Eq. 87-88).
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    rng = np.random.default_rng(5)
    g = rng.standard_normal(SIGNAL_SIZE)
    h = rng.standard_normal(SIGNAL_SIZE)
    G = impl._apply(kernel, g)
    H = impl._apply(kernel, h)

    zero = np.zeros(SIGNAL_SIZE, dtype=complex)
    convolution = sum(
        (g[k0] * _generalized_shift(kernel, H, k0) for k0 in range(SIGNAL_SIZE)), zero
    )
    swapped = sum(
        (h[k0] * _generalized_shift(kernel, G, k0) for k0 in range(SIGNAL_SIZE)), zero
    )
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(convolution, swapped, rtol=tol, atol=tol)

    lhs = impl._apply(kernel, convolution)
    rhs = G * H
    np.testing.assert_allclose(lhs, rhs, rtol=tol, atol=tol)


def test_dht_multiplication_rule(implementation, order):
    """Multiplication in space is a generalized convolution in frequency (Eq. 89-92).

    ``f = g * h`` (entrywise, space domain) transforms to
    ``sum_q G[q] * shift(h, q)[m]`` -- the mirror of the convolution rule
    above, with the roles of space and frequency swapped. Also checks this
    frequency-domain convolution commutes.
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = impl._bessel_kernel(order, SIGNAL_SIZE)
    rng = np.random.default_rng(6)
    g = rng.standard_normal(SIGNAL_SIZE)
    h = rng.standard_normal(SIGNAL_SIZE)
    G = impl._apply(kernel, g)

    f = g * h
    F = impl._apply(kernel, f)
    zero = np.zeros(SIGNAL_SIZE, dtype=complex)
    freq_convolution = sum(
        (G[q] * _generalized_shift(kernel, h, q) for q in range(SIGNAL_SIZE)), zero
    )
    tol = dht_tolerance(order, SIGNAL_SIZE)
    np.testing.assert_allclose(F, freq_convolution, rtol=tol, atol=tol)

    H = impl._apply(kernel, h)
    swapped = sum(
        (H[q] * _generalized_shift(kernel, g, q) for q in range(SIGNAL_SIZE)), zero
    )
    np.testing.assert_allclose(freq_convolution, swapped, rtol=tol, atol=tol)
