"""The discrete Hankel transform (DHT).

Implements the DHT as a transform in its own right (not a discretized integral),
following N. Baddour, "The Discrete Hankel Transform" (2019,
``.local_files/sources/baddour2019.md``). ``hankel_transform`` and
``inverse_hankel_transform`` are the public entry points; each dispatches to
one of several interchangeable ``DHTImplementation`` strategies (Strategy
pattern, see ``pypft.dht._base.BaseDHT``), selected by the ``implementation``
argument and defaulting to ``DEFAULT_IMPLEMENTATION``.
"""

from enum import Enum, auto

import numpy as np

from pypft.utils.validators import (
    EnumValidator,
    FloatValidator,
    IntValidator,
    NumpyValidator,
)

from ._base import BaseDHT
from ._cached import CachedBesselDHT
from ._naive import NaiveDHT
from ._vectorized import VectorizedDHT


class DHTImplementation(Enum):
    """Selectable strategies for computing the discrete Hankel transform."""

    NAIVE = auto()
    CACHED_BESSEL = auto()
    VECTORIZED = auto()


_IMPLEMENTATIONS: dict[DHTImplementation, type[BaseDHT]] = {
    DHTImplementation.NAIVE: NaiveDHT,
    DHTImplementation.CACHED_BESSEL: CachedBesselDHT,
    DHTImplementation.VECTORIZED: VectorizedDHT,
}

DEFAULT_IMPLEMENTATION: DHTImplementation = DHTImplementation.CACHED_BESSEL
"""The implementation used when ``implementation`` is not given explicitly.

Hardcoded to the fastest *numerically sound* implementation found by
``.local_files/benchmarks/run_dht_benchmarks.py``: for repeated forward calls at a
fixed order/size (the realistic usage pattern, e.g. many radial lines sharing one
discretization), ``CACHED_BESSEL`` is ~3000x faster than ``NAIVE``. The
previously-default ``RECURRENCE_BESSEL`` matched that speed but was removed: its
upward Bessel-order recurrence is exponentially unstable once the order exceeds
the argument, which the kernel does by construction, so it silently diverged
above order ~12 (measured ``max|Y @ Y - I|`` reaching ``2.1e+16`` by order 47).
``VECTORIZED`` remains slower than plain BLAS matmul at the benchmarked sizes,
now inheriting ``CACHED_BESSEL``'s kernel instead.
"""


def _validate_transform_inputs(
    signal: np.ndarray,
    n: int,
    R: float,
    implementation: DHTImplementation,
    axis: int,
) -> None:
    """Validate the shared arguments of the forward/inverse DHT entry points.

    :param signal: The signal to be transformed.
    :type signal: np.ndarray
    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param R: The space-domain limit the signal is assumed confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :param axis: The axis of ``signal`` holding the length-``size`` samples.
    :type axis: int
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(signal)
    NumpyValidator.value_is_at_least_1d(signal)
    IntValidator.type_is_int(n)
    IntValidator.value_is_non_negative(n)
    FloatValidator.type_is_float(R)
    FloatValidator.value_is_positive(R)
    EnumValidator.type_is_enum(implementation)
    EnumValidator.value_is_enum_member(implementation, DHTImplementation)
    IntValidator.type_is_int(axis)
    NumpyValidator.value_has_axis(signal, axis)


def hankel_transform(
    f: np.ndarray,
    n: int,
    R: float,
    implementation: DHTImplementation = DEFAULT_IMPLEMENTATION,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Compute the forward discrete Hankel transform of order ``n``.

    :param f: The space-domain samples ``f(r_nk)``, sampled at the points
        returned by ``pypft.dht._base.BaseDHT.sample_points``, along ``axis``
        of an otherwise arbitrary-rank array.
    :type f: np.ndarray
    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param R: The space-domain limit ``f`` is assumed confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :param axis: The axis of ``f`` holding the length-``size`` samples.
        Keyword-only and last so every pre-existing 4-positional-argument call
        site is unaffected.
    :type axis: int
    :returns: The frequency-domain samples ``F(rho_nk)``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(f, n, R, implementation, axis)
    return _IMPLEMENTATIONS[implementation].forward(f, n, R, axis=axis)


def inverse_hankel_transform(
    F: np.ndarray,
    n: int,
    R: float,
    implementation: DHTImplementation = DEFAULT_IMPLEMENTATION,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Compute the inverse discrete Hankel transform of order ``n``.

    :param F: The frequency-domain samples ``F(rho_nk)``, along ``axis`` of an
        otherwise arbitrary-rank array.
    :type F: np.ndarray
    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param R: The space-domain limit the reconstructed signal is confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :param axis: The axis of ``F`` holding the length-``size`` samples.
        Keyword-only and last so every pre-existing 4-positional-argument call
        site is unaffected.
    :type axis: int
    :returns: The space-domain samples ``f(r_nk)``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(F, n, R, implementation, axis)
    return _IMPLEMENTATIONS[implementation].inverse(F, n, R, axis=axis)


def sample_points(
    n: int,
    size: int,
    R: float,
    implementation: DHTImplementation = DEFAULT_IMPLEMENTATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the discretization's space- and frequency-domain sample points.

    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param size: The number of samples (``N - 1`` in baddour2019.md's notation).
    :type size: int
    :param R: The space-domain limit the signal is assumed confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :returns: The sample points ``r_nk`` and ``rho_nk``, each of length ``size``.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    IntValidator.type_is_int(n)
    IntValidator.value_is_non_negative(n)
    IntValidator.type_is_int(size)
    IntValidator.value_is_non_negative(size)
    FloatValidator.type_is_float(R)
    FloatValidator.value_is_positive(R)
    EnumValidator.type_is_enum(implementation)
    EnumValidator.value_is_enum_member(implementation, DHTImplementation)
    return _IMPLEMENTATIONS[implementation].sample_points(n, size, R)
