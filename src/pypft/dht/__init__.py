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
from ._recurrence import RecurrenceBesselDHT
from ._vectorized import VectorizedDHT


class DHTImplementation(Enum):
    """Selectable strategies for computing the discrete Hankel transform."""

    NAIVE = auto()
    CACHED_BESSEL = auto()
    RECURRENCE_BESSEL = auto()
    VECTORIZED = auto()


_IMPLEMENTATIONS: dict[DHTImplementation, type[BaseDHT]] = {
    DHTImplementation.NAIVE: NaiveDHT,
    DHTImplementation.CACHED_BESSEL: CachedBesselDHT,
    DHTImplementation.RECURRENCE_BESSEL: RecurrenceBesselDHT,
    DHTImplementation.VECTORIZED: VectorizedDHT,
}

DEFAULT_IMPLEMENTATION: DHTImplementation = DHTImplementation.RECURRENCE_BESSEL
"""The implementation used when ``implementation`` is not given explicitly.

Hardcoded to the fastest implementation found by
``.local_files/benchmarks/run_dht_benchmarks.py``: for repeated forward calls at a
fixed order/size (the realistic usage pattern, e.g. many radial lines sharing one
discretization), ``RECURRENCE_BESSEL`` and ``CACHED_BESSEL`` are statistically
tied and ~3000x faster than ``NAIVE``, while ``VECTORIZED``'s ``numba`` thread
overhead loses to plain BLAS matmul at the benchmarked sizes.
"""


def _validate_transform_inputs(
    signal: np.ndarray, n: int, R: float, implementation: DHTImplementation
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
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(signal)
    NumpyValidator.value_is_1d(signal)
    IntValidator.type_is_int(n)
    IntValidator.value_is_non_negative(n)
    FloatValidator.type_is_float(R)
    FloatValidator.value_is_positive(R)
    EnumValidator.type_is_enum(implementation)
    EnumValidator.value_is_enum_member(implementation, DHTImplementation)


def hankel_transform(
    f: np.ndarray,
    n: int,
    R: float,
    implementation: DHTImplementation = DEFAULT_IMPLEMENTATION,
) -> np.ndarray:
    """Compute the forward discrete Hankel transform of order ``n``.

    :param f: The space-domain samples ``f(r_nk)``, sampled at the points
        returned by ``pypft.dht._base.BaseDHT.sample_points``.
    :type f: np.ndarray
    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param R: The space-domain limit ``f`` is assumed confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :returns: The frequency-domain samples ``F(rho_nk)``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(f, n, R, implementation)
    return _IMPLEMENTATIONS[implementation].forward(f, n, R)


def inverse_hankel_transform(
    F: np.ndarray,
    n: int,
    R: float,
    implementation: DHTImplementation = DEFAULT_IMPLEMENTATION,
) -> np.ndarray:
    """Compute the inverse discrete Hankel transform of order ``n``.

    :param F: The frequency-domain samples ``F(rho_nk)``.
    :type F: np.ndarray
    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param R: The space-domain limit the reconstructed signal is confined to.
    :type R: float
    :param implementation: The DHT implementation strategy to use.
    :type implementation: DHTImplementation
    :returns: The space-domain samples ``f(r_nk)``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(F, n, R, implementation)
    return _IMPLEMENTATIONS[implementation].inverse(F, n, R)


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
