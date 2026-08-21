"""The angular discrete Fourier transform (DFT).

Implements the centered angular DFT/IDFT that sits between PyPFT's stored,
centered-angular arrays and the discrete Hankel transform's per-harmonic
processing -- the ``FFT_phi``/``IFFT_theta`` steps of ``README.md``'s PFT
chain. ``angular_dft`` and ``inverse_angular_dft`` are the public entry
points; each dispatches to one of several interchangeable
``DFTImplementation`` strategies (Strategy pattern, see
``pypft.dft._base.BaseDFT``), selected by the ``implementation`` argument and
defaulting to ``DEFAULT_IMPLEMENTATION``.

This module also owns the harmonic-range derivation (``harmonics``) and the
angular-parity classification (``AngularParity``/``angular_parity``): both
sample-count parities are numerically valid, so parity is reported, never
validated against.
"""

from enum import Enum, auto

import numpy as np

from pypft.utils.validators import EnumValidator, IntValidator, NumpyValidator

from ._base import BaseDFT
from ._numpy import NumpyDFT
from ._scipy import ScipyDFT


class DFTImplementation(Enum):
    """Selectable strategies for computing the angular DFT."""

    NUMPY = auto()
    SCIPY = auto()


_IMPLEMENTATIONS: dict[DFTImplementation, type[BaseDFT]] = {
    DFTImplementation.NUMPY: NumpyDFT,
    DFTImplementation.SCIPY: ScipyDFT,
}

DEFAULT_IMPLEMENTATION: DFTImplementation = DFTImplementation.NUMPY
"""The implementation used when ``implementation`` is not given explicitly.

Hardcoded to the fastest implementation found by
``benchmarks/run_dft_benchmarks.py`` for repeated forward calls at a fixed
size (the realistic usage pattern: one angular DFT per radial line, many
lines sharing one size) -- ``NUMPY`` at ~11.7us vs. ``SCIPY``'s ~12.1us,
essentially tied. ``SCIPY`` wins by a wide margin (~33%) on a batched,
non-trailing-axis input instead (~2.8ms vs. ~4.2ms at a (32, 128, 64) shape),
but that is not the scenario this default is chosen for -- see
``benchmarks/bench_dft.py``'s ``test_bench_forward_batched``. A separate
``SCIPY_WORKERS`` (``workers=-1``) implementation was considered for that
batched case and rejected: explicit worker parallelism measured only ~2%
faster than ``SCIPY``'s own default there, so the gain is ``scipy.fft``'s
algorithm, not parallelism -- not worth a third strategy. See
``.local_files/benchmarks/results/`` for the full report.
"""


class AngularParity(Enum):
    """Whether an angular sample count is even or odd.

    A derived, *reported* property of a size, never a precondition: both
    parities are numerically valid, so no validator rejects an odd or even
    ``n_angular`` -- see ``angular_parity``.
    """

    ODD = auto()
    EVEN = auto()


def angular_parity(n_angular: int) -> AngularParity:
    """Classify an angular sample count's parity.

    :param n_angular: The number of angular samples.
    :type n_angular: int
    :returns: ``AngularParity.EVEN`` if ``n_angular`` is even, else ``ODD``.
    :rtype: AngularParity
    :raises TypeError: If ``n_angular`` is not an int.
    :raises ValueError: If ``n_angular`` is not strictly positive.

    """
    IntValidator.type_is_int(value=n_angular)
    IntValidator.value_is_positive(value=n_angular)
    return AngularParity.EVEN if n_angular % 2 == 0 else AngularParity.ODD


def harmonics(n_angular: int) -> np.ndarray:
    """Compute the centered harmonic indices for ``n_angular`` samples.

    Correct for either parity: index ``i`` holds harmonic
    ``i - n_angular // 2``, matching ``pypft.axes``'s centering convention
    (and, transitively, what ``angular_dft``'s output means at each index)
    exactly.

    :param n_angular: The number of angular samples.
    :type n_angular: int
    :returns: The harmonic indices, ascending, length ``n_angular``.
    :rtype: np.ndarray
    :raises TypeError: If ``n_angular`` is not an int.
    :raises ValueError: If ``n_angular`` is not strictly positive.

    """
    IntValidator.type_is_int(value=n_angular)
    IntValidator.value_is_positive(value=n_angular)
    return np.arange(start=-(n_angular // 2), stop=n_angular - n_angular // 2)


def _validate_transform_inputs(
    signal: np.ndarray, implementation: DFTImplementation, axis: int
) -> None:
    """Validate the shared arguments of the forward/inverse DFT entry points.

    :param signal: The signal to be transformed.
    :type signal: np.ndarray
    :param implementation: The DFT implementation strategy to use.
    :type implementation: DFTImplementation
    :param axis: The angular axis of ``signal``.
    :type axis: int
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(value=signal)
    NumpyValidator.value_is_at_least_1d(value=signal)
    EnumValidator.type_is_enum(value=implementation)
    EnumValidator.value_is_enum_member(value=implementation, enum_class=DFTImplementation)
    IntValidator.type_is_int(value=axis)
    NumpyValidator.value_has_axis(value=signal, axis=axis)


def angular_dft(
    x: np.ndarray,
    implementation: DFTImplementation = DEFAULT_IMPLEMENTATION,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Compute the centered forward angular discrete Fourier transform.

    :param x: The centered space-domain samples, along ``axis`` of an
        otherwise arbitrary-rank array.
    :type x: np.ndarray
    :param implementation: The DFT implementation strategy to use.
    :type implementation: DFTImplementation
    :param axis: The angular axis of ``x``. Keyword-only, matching
        ``pypft.dht.hankel_transform``'s convention.
    :type axis: int
    :returns: The centered harmonic-domain coefficients.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(signal=x, implementation=implementation, axis=axis)
    return _IMPLEMENTATIONS[implementation].forward(x=x, axis=axis)


def inverse_angular_dft(
    X: np.ndarray,
    implementation: DFTImplementation = DEFAULT_IMPLEMENTATION,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Compute the centered inverse angular discrete Fourier transform.

    :param X: The centered harmonic-domain coefficients, along ``axis`` of
        an otherwise arbitrary-rank array.
    :type X: np.ndarray
    :param implementation: The DFT implementation strategy to use.
    :type implementation: DFTImplementation
    :param axis: The angular axis of ``X``. Keyword-only, matching
        ``pypft.dht.inverse_hankel_transform``'s convention.
    :type axis: int
    :returns: The centered space-domain samples.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    _validate_transform_inputs(signal=X, implementation=implementation, axis=axis)
    return _IMPLEMENTATIONS[implementation].inverse(X=X, axis=axis)
