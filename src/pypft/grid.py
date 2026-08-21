"""The transform's own sampling grid: order-dependent, non-uniform, Bessel-zero based.

Baddour's discrete Hankel transform (``pypft.dht``) does not sample a radial profile
uniformly: each harmonic order ``n`` has its own radial sample positions ``r_nk``, tied
to the zeros of ``J_n``. Composed with the angular DFT (``pypft.dft``), this means every
*angular* index of a stored polar array carries its own radial grid, since the angular
DFT/IDFT step is exactly what turns a "physical angle" index into a "harmonic order"
index and back. ``PolarGrid`` is the frozen, hashable value object describing that grid
for a given ``(n_radial, n_angular, R)``; ``sample_cartesian`` is the production sampler
that resamples an ordinary image onto it (unlike
``pypft.geometry.cartesian_to_polar``, which resamples onto a uniform grid purely for
illustration); ``check_adequacy`` and ``check_nyquist_adequacy`` are empirical/
analytical guards that warn when a grid's angular and radial sample counts are
mismatched, since that mismatch degrades accuracy without ever raising an error on its
own.

The angular axis is one-dimensional and shared by both the space and frequency domains
(``theta``/``psi``): index ``i`` sits at physical angle/harmonic
``harmonics(n_angular)[i] * 2 * pi / n_angular``, i.e. the same centered convention
every other stored PyPFT array uses, so a grid's row index lines up with the
corresponding row of a centered angular array without any reordering at this boundary.
"""

import warnings
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from scipy.special import jn_zeros

from pypft.dft import AngularParity, angular_parity
from pypft.dft import harmonics as _angular_harmonics
from pypft.dht import sample_points
from pypft.utils.validators import (
    EnumValidator,
    FloatValidator,
    IntValidator,
    NumpyValidator,
)

# ======================================================================================
# The grid's own enum and warning types
# ======================================================================================


class LimitKind(Enum):
    """Whether a ``PolarGrid`` assumes a space-limited or a band-limited function.

    A space-limited function is confined to ``r <= R`` in the space domain (PeerJ CS
    Part II, Eqs. 14-15); a band-limited function is instead confined to ``rho <= R``
    in the frequency domain (Eqs. 16-17), with ``R`` then read as the band limit
    ``Wr``. The two cases use the exact same Bessel-zero ratios, just with the roles of
    ``r`` and ``rho`` swapped -- see ``PolarGrid.r``/``PolarGrid.rho``.
    """

    SPACE_LIMITED = auto()
    BAND_LIMITED = auto()


class AdequacyWarning(UserWarning):
    """Raised when a grid's ``n_radial`` is likely too small for its ``n_angular``."""


class NyquistWarning(UserWarning):
    """Raised when a grid violates the discrete Hankel transform's Nyquist condition."""


# ======================================================================================
# The grid itself
# ======================================================================================


@dataclass(frozen=True)
class PolarGrid:
    """The discrete Hankel transform's own ``(radial, angular)`` sampling grid.

    Frozen and hashable (the default dataclass hash, since every field is itself
    hashable) so a grid can key a kernel cache the way ``pypft.dht``'s ``(n, size)``
    pair already does. Every array-valued attribute is a property computed from the
    four stored fields on access rather than cached on the instance, since the
    underlying ``pypft.dht.sample_points`` call is already memoized by its own
    ``(order, size)`` kernel cache -- recomputing here is cheap, and keeps this class a
    plain, comparable value object.

    :param n_radial: The number of radial samples (Baddour's ``N - 1``).
    :type n_radial: int
    :param n_angular: The number of angular samples.
    :type n_angular: int
    :param R: The space limit (or, for ``LimitKind.BAND_LIMITED``, the band limit
        ``Wr``) the sampled function is assumed confined to.
    :type R: float
    :param limit_kind: Whether ``R`` names a space limit or a band limit.
    :type limit_kind: LimitKind
    :raises TypeError: If any field has the wrong type.
    :raises ValueError: If any field has an invalid value.

    """

    n_radial: int
    n_angular: int
    R: float
    limit_kind: LimitKind = LimitKind.SPACE_LIMITED

    def __post_init__(self) -> None:
        """Validate every field once, right after construction."""
        IntValidator.type_is_int(self.n_radial)
        IntValidator.value_is_positive(self.n_radial)
        IntValidator.type_is_int(self.n_angular)
        IntValidator.value_is_positive(self.n_angular)
        FloatValidator.type_is_float(self.R)
        FloatValidator.value_is_positive(self.R)
        EnumValidator.type_is_enum(self.limit_kind)
        EnumValidator.value_is_enum_member(self.limit_kind, LimitKind)

    @property
    def harmonics(self) -> np.ndarray:
        """The centered harmonic index at each angular row, length ``n_angular``."""
        return _angular_harmonics(self.n_angular)

    @property
    def parity(self) -> AngularParity:
        """Whether ``n_angular`` is even or odd."""
        return angular_parity(self.n_angular)

    @property
    def theta(self) -> np.ndarray:
        """The centered angular sample points, shared by both domains.

        Length ``n_angular``.
        """
        return self.harmonics * (2.0 * np.pi / self.n_angular)

    @property
    def psi(self) -> np.ndarray:
        """The frequency domain's angular sample points -- identical to ``theta``."""
        return self.theta

    def _radial_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the space- and frequency-domain radial grids, one row per harmonic.

        :returns: The space-domain and frequency-domain radial grids, each
            ``(n_angular, n_radial)``.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        space = np.empty((self.n_angular, self.n_radial))
        frequency = np.empty((self.n_angular, self.n_radial))
        for row, order in enumerate(self.harmonics):
            # Each row's order is the harmonic's own |n| -- the DHT kernel only
            # ever depends on the order's magnitude (Y^{(-n)N} = (-1)^n Y^{nN}).
            space[row, :], frequency[row, :] = sample_points(
                int(abs(order)), self.n_radial, self.R
            )
        return space, frequency

    @property
    def r(self) -> np.ndarray:
        """The space-domain radial grid, ``(n_angular, n_radial)``."""
        space, frequency = self._radial_grids()
        return space if self.limit_kind is LimitKind.SPACE_LIMITED else frequency

    @property
    def rho(self) -> np.ndarray:
        """The frequency-domain radial grid, ``(n_angular, n_radial)``.

        For ``LimitKind.BAND_LIMITED``, this and ``r`` swap which of the two
        ``sample_points`` outputs they return -- PeerJ CS Part II notes the
        band-limited grid has "the same shape...but the domains are reversed"
        relative to the space-limited one.

        """
        space, frequency = self._radial_grids()
        return frequency if self.limit_kind is LimitKind.SPACE_LIMITED else space


def _type_is_polar_grid(value: PolarGrid) -> None:
    """Type-validator for ``PolarGrid``, defined here since the type is defined here.

    :param value: The value to be validated.
    :type value: PolarGrid
    :raises TypeError: If the value is not a ``PolarGrid``.

    """
    if not isinstance(value, PolarGrid):
        raise TypeError(f"value must be PolarGrid, got {type(value).__name__}")


# ======================================================================================
# The production sampler
# ======================================================================================


def sample_cartesian(image: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Resample a Cartesian image onto ``grid``'s own, non-uniform sample points.

    Unlike ``pypft.geometry.cartesian_to_polar`` (a *uniform* radial axis, used there
    only as the first illustration of what "polar" means for an image), this is the
    sampler that actually feeds the transform: a single ``cv2.remap`` call at
    ``grid.r``/``grid.theta``, using the same angle convention as ``pypft.geometry``
    (measured directly on image coordinates, with no ``y``-flip).

    :param image: A ``(height, width)`` grayscale Cartesian image.
    :type image: np.ndarray
    :param grid: The grid to sample onto.
    :type grid: PolarGrid
    :returns: A ``(n_angular, n_radial)`` array of ``image`` resampled at ``grid.r``,
        ``grid.theta``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(image)
    NumpyValidator.value_is_2d(image)
    NumpyValidator.value_is_finite(image)
    _type_is_polar_grid(grid)

    height, width = image.shape
    center_x, center_y = width / 2.0, height / 2.0
    # grid.theta is 1-D (n_angular,); broadcasting it against grid.r's own
    # (n_angular, n_radial) shape gives every row its own angle, matching how the
    # angular index also selects the row's Bessel order in grid.r itself.
    map_x = (center_x + grid.r * np.cos(grid.theta)[:, np.newaxis]).astype(np.float32)
    map_y = (center_y + grid.r * np.sin(grid.theta)[:, np.newaxis]).astype(np.float32)
    return cv2.remap(image.astype(np.float64), map_x, map_y, cv2.INTER_LINEAR)


# ======================================================================================
# Adequacy and Nyquist guards -- warnings, never errors
# ======================================================================================

#: Coefficients of a log-log least-squares fit of the forward Gaussian oracle's
#: average dB error to ``(n_angular, n_radial)``, fitted from nine measured
#: ``(n_angular, n_radial, E_avg)`` points spanning ``n_angular in (15, 32, 64)`` and
#: ``n_radial in (383, 767, 1535)`` at ``R = 40``. The fit's residual is under 0.6 dB
#: at every measured point, confirming the relationship is close to log-linear and
#: separable in this regime: each doubling of ``n_radial`` improves ``E_avg`` by about
#: 24.5 dB, and each doubling of ``n_angular`` worsens it by about 10.8 dB.
_ADEQUACY_INTERCEPT_DB = 104.23
_ADEQUACY_N_ANGULAR_COEFFICIENT_DB = 10.82
_ADEQUACY_N_RADIAL_COEFFICIENT_DB = -24.51

#: The predicted forward ``E_avg`` (dB) below which a grid is considered adequate --
#: chosen to match the PFT pipeline's own forward-accuracy acceptance threshold, so
#: the two stay consistent.
_ADEQUACY_THRESHOLD_DB = -60.0


def _predicted_forward_error_db(n_angular: int, n_radial: int) -> float:
    """Predict the forward PFT's average dB error for a given grid size.

    :param n_angular: The number of angular samples.
    :type n_angular: int
    :param n_radial: The number of radial samples.
    :type n_radial: int
    :returns: The predicted average dB error.
    :rtype: float

    """
    return (
        _ADEQUACY_INTERCEPT_DB
        + _ADEQUACY_N_ANGULAR_COEFFICIENT_DB * np.log2(n_angular)
        + _ADEQUACY_N_RADIAL_COEFFICIENT_DB * np.log2(n_radial)
    )


def check_adequacy(grid: PolarGrid) -> None:
    """Warn if ``grid``'s ``n_radial`` is likely too small for its ``n_angular``.

    Raising angular resolution alone destroys accuracy: doubling ``n_angular``
    without growing ``n_radial`` to match costs about 11 dB, to the point that a
    high-``n_angular``, modest-``n_radial`` grid's *maximum* dB error can turn
    positive -- meaning the reconstruction is worse than useless at its worst point.
    This uses the measured relation (see ``_ADEQUACY_INTERCEPT_DB``'s docstring) to
    predict the average error and warns, with a suggested ``n_radial``, whenever that
    prediction is worse than the pipeline's own accuracy target.

    :param grid: The grid to check.
    :type grid: PolarGrid
    :raises TypeError: If ``grid`` is not a ``PolarGrid``.

    """
    _type_is_polar_grid(grid)
    predicted = _predicted_forward_error_db(grid.n_angular, grid.n_radial)
    if predicted > _ADEQUACY_THRESHOLD_DB:
        needed_log2_n_radial = (
            _ADEQUACY_INTERCEPT_DB
            + _ADEQUACY_N_ANGULAR_COEFFICIENT_DB * np.log2(grid.n_angular)
            - _ADEQUACY_THRESHOLD_DB
        ) / -_ADEQUACY_N_RADIAL_COEFFICIENT_DB
        suggested_n_radial = int(np.ceil(2.0**needed_log2_n_radial))
        warnings.warn(
            f"n_radial={grid.n_radial} is likely inadequate for "
            f"n_angular={grid.n_angular}: predicted forward average error is "
            f"{predicted:.1f} dB, worse than the {_ADEQUACY_THRESHOLD_DB:.0f} dB "
            f"target; try n_radial >= {suggested_n_radial}.",
            AdequacyWarning,
            stacklevel=2,
        )


def check_nyquist_adequacy(grid: PolarGrid, band_limit: float) -> None:
    """Warn if ``grid`` violates the discrete Hankel transform's Nyquist condition.

    PeerJ CS Part II's Eq. 21 requires ``j_(0, N1) >= band_limit * R``, using the
    zero-order zero specifically because ``j_(0, N1)`` is the smallest of the zeros
    ``j_(-M, N1), ..., j_(0, N1), ..., j_(M, N1)`` the transform actually uses, making
    it the binding constraint across every harmonic order. Only
    ``LimitKind.SPACE_LIMITED`` is supported: the band-limited case swaps the roles of
    ``R`` and ``band_limit`` in a way this helper does not attempt to generalize.

    :param grid: The space-limited grid to check.
    :type grid: PolarGrid
    :param band_limit: The band limit ``Wr`` the sampled function is assumed confined
        to.
    :type band_limit: float
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``band_limit`` is not strictly positive.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    _type_is_polar_grid(grid)
    FloatValidator.type_is_float(band_limit)
    FloatValidator.value_is_positive(band_limit)
    if grid.limit_kind is not LimitKind.SPACE_LIMITED:
        raise NotImplementedError(
            "check_nyquist_adequacy only supports LimitKind.SPACE_LIMITED grids"
        )

    # jn_zeros(0, N) returns the first N zeros of J_0; the N1-th one (Baddour's
    # notation) is the (n_radial + 1)-th zero, since n_radial itself is "N1 - 1".
    j_0_n1 = jn_zeros(0, grid.n_radial + 1)[-1]
    required = band_limit * grid.R
    if j_0_n1 < required:
        suggested_n_radial = int(np.ceil(2.0 * band_limit * grid.R / np.pi))
        warnings.warn(
            f"grid violates the DHT Nyquist condition: j_(0,{grid.n_radial + 1})="
            f"{j_0_n1:.3f} < band_limit*R={required:.3f}; try n_radial >= "
            f"{suggested_n_radial} (Eq. 24's asymptotic estimate).",
            NyquistWarning,
            stacklevel=2,
        )
