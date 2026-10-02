"""The transform's own sampling grid: order-dependent, non-uniform, Bessel-zero based.

``PolarGrid`` is the frozen, hashable value object describing the grid of PeerJ CS
Part II, Eqs. 14-17, for a given ``(n_radial, n_angular, R)``. Its angles are uniform
and centered; its radii are not. In the space domain, row ``i`` of ``PolarGrid.r`` is
the spoke at angle ``theta_i``, and its radii are ``r_pk = j_{|p|k} R / j_{|p|N1}``,
where ``p = harmonics(n_angular)[i]`` is the spoke's own *angular sample index* and
``j_{|p|k}`` the ``k``-th zero of ``J_{|p|}``. Harmonics only exist after the angular
DFT. The transform's derivation identifies the angular sample index with the harmonic
order (Mathematics Part I, Eq. 18, a "key assumption" of the development; Appendix A.6
compares it with the conventional alternative, Eq. A17), which is what makes the
discrete transform square and exactly invertible.

Two consequences matter to anyone sampling data for it:

- **A row of constant radial index is not a ring.** Every spoke uses its own radii,
  so the same index ``k`` sits at a different radius on different spokes, most of all
  near the center. A uniform polar array (``pypft.geometry.cartesian_to_polar``) has
  the same spokes but the same radii on every spoke: it is *not* on this grid, and
  must be interpolated along each spoke onto ``PolarGrid.r`` before it is
  transformed as an approximation of the continuous Fourier transform.
- **On-grid samples still carry an approximation error** from the identification
  itself, which the average dB error ``E_avg`` hides; the relative L2 error shows it.

``DESIGN_NOTES.md``, "Grid: the spatial row index is a spoke, and the transform
identifies it with a harmonic," has the measurements behind both. Two functions put
data on this grid: ``sample_cartesian`` evaluates an ordinary image at the grid's
points, and ``resample_uniform_polar`` interpolates uniform polar data (a
``cartesian_to_polar`` result, or an acquisition with equally spaced samples along
each spoke) along each spoke onto them. ``pypft.ring`` samples on true rings instead,
for the ring-consistent route that avoids the identification. ``check_adequacy`` and
``check_nyquist_adequacy`` are empirical/analytical guards that warn when a grid's
angular and radial sample counts are mismatched, since that mismatch degrades accuracy
without ever raising an error on its own.

The angular axis is one-dimensional and shared by both the space and frequency domains
(``theta``/``psi``): index ``i`` sits at physical angle
``harmonics(n_angular)[i] * 2 * pi / n_angular``, i.e. the same centered convention
every other stored PyPFT array uses, so a grid's row index lines up with the
corresponding column of a centered angular array without any reordering at this
boundary.
"""

import warnings
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jn_zeros

from pypft.axes import PolarAxis, _value_is_polar_sample_or_batch
from pypft.dft import AngularParity, angular_parity
from pypft.dft import harmonics as _angular_harmonics
from pypft.dht import sample_points
from pypft.utils.validators import (
    EnumValidator,
    FloatValidator,
    IntValidator,
    NumpyValidator,
)

#: The fewest uniform radial samples a cubic spline can be fitted through.
_MIN_UNIFORM_RADIAL_SAMPLES = 2

# ======================================================================================
# The grid's own enum, warning and error types
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


class AngularUpsamplingWarning(UserWarning):
    """Raised when uniform polar data has fewer spokes than the grid it is put on."""


class RadiusCoverageError(ValueError):
    """Raised when a grid reaches past the radius that uniform polar data covers."""


# ======================================================================================
# The grid itself
# ======================================================================================


@dataclass(frozen=True)
class PolarGrid:
    """The polar Fourier transform's own sampling grid.

    Row ``i`` of ``r`` (space domain) and ``rho`` (frequency domain) is the spoke at
    angle ``theta[i]``, with radii from the zeros of ``J_{|p|}``, where
    ``p = harmonics[i]`` is that spoke's angular sample index. The transform
    identifies ``p`` with the harmonic order (see the module docstring), so the same
    radial index sits at a different radius on each spoke: a uniform polar array is
    not on this grid and must be interpolated along each spoke onto ``r`` first.

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
        IntValidator.type_is_int(value=self.n_radial)
        IntValidator.value_is_positive(value=self.n_radial)
        IntValidator.type_is_int(value=self.n_angular)
        IntValidator.value_is_positive(value=self.n_angular)
        FloatValidator.type_is_float(value=self.R)
        FloatValidator.value_is_positive(value=self.R)
        EnumValidator.type_is_enum(value=self.limit_kind)
        EnumValidator.value_is_enum_member(value=self.limit_kind, enum_class=LimitKind)

    @property
    def harmonics(self) -> np.ndarray:
        """The centered angular sample index of each row, length ``n_angular``.

        In the space domain this indexes spokes, not harmonics: ``harmonics[i]`` is
        the index ``p`` whose Bessel order ``|p|`` places row ``i``'s radii. It names
        the harmonic order only after the angular DFT, which the transform
        identifies with ``p``. The name follows that identification.
        """
        return _angular_harmonics(self.n_angular)

    @property
    def parity(self) -> AngularParity:
        """Whether ``n_angular`` is even or odd."""
        return angular_parity(self.n_angular)

    @property
    def theta(self) -> np.ndarray:
        """The centered angle of each spoke, shared by both domains.

        Length ``n_angular``, uniform, ``harmonics * 2 * pi / n_angular``. A uniform
        polar grid with the same ``n_angular`` has exactly these spokes; only the
        radii along them differ.
        """
        return self.harmonics * (2.0 * np.pi / self.n_angular)

    @property
    def psi(self) -> np.ndarray:
        """The frequency domain's angular sample points -- identical to ``theta``."""
        return self.theta

    def _radial_grids(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the space- and frequency-domain radial grids, one row per spoke.

        :returns: The space-domain and frequency-domain radial grids, each
            ``(n_angular, n_radial)``.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        space = np.empty((self.n_angular, self.n_radial))
        frequency = np.empty((self.n_angular, self.n_radial))
        for row, order in enumerate(self.harmonics):
            # Each spoke's Bessel order is |p| of its own angular sample index --
            # the DHT kernel only ever depends on the order's magnitude
            # (Y^{(-n)N} = (-1)^n Y^{nN}).
            space[row, :], frequency[row, :] = sample_points(
                n=int(abs(order)), size=self.n_radial, R=self.R
            )
        return space, frequency

    @property
    def r(self) -> np.ndarray:
        """The space-domain radii, ``(n_angular, n_radial)``: row ``i`` is one spoke.

        Row ``i`` holds ``r_pk = j_{|p|k} R / j_{|p|N1}``, ``k = 1 .. n_radial``,
        for ``p = harmonics[i]`` (space-limited case). Rows ``p`` and ``-p`` are
        equal; every other pair differs, so a column of constant radial index is not
        a ring. Near the outer edge the spacing along row ``i`` settles to about
        ``pi * R / j_{|p|N1}``, close to uniform; near the center it is not, and the
        innermost radius grows with ``|p|``.
        """
        space, frequency = self._radial_grids()
        return space if self.limit_kind is LimitKind.SPACE_LIMITED else frequency

    @property
    def rho(self) -> np.ndarray:
        """The frequency-domain radii, ``(n_angular, n_radial)``: row ``i`` is a spoke.

        Row ``i`` holds ``rho_pm = j_{|p|m} / R`` for ``p = harmonics[i]``
        (space-limited case), at angle ``psi[i]``: a forward PFT output entry
        approximates the continuous transform at that point. For
        ``LimitKind.BAND_LIMITED``, this and ``r`` swap which of the two
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
    sampler that actually feeds the transform -- each spoke is evaluated at its own
    radii, so the result is on the grid rather than on uniform rings: a single
    ``cv2.remap`` call at
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
    NumpyValidator.type_is_ndarray(value=image)
    NumpyValidator.value_is_2d(value=image)
    NumpyValidator.value_is_finite(value=image)
    _type_is_polar_grid(value=grid)

    height, width = image.shape
    center_x, center_y = width / 2.0, height / 2.0
    # grid.theta is 1-D (n_angular,); broadcasting it against grid.r's own
    # (n_angular, n_radial) shape gives every row its own angle, matching how the
    # angular index also selects the row's Bessel order in grid.r itself.
    map_x = (center_x + grid.r * np.cos(grid.theta)[:, np.newaxis]).astype(np.float32)
    map_y = (center_y + grid.r * np.sin(grid.theta)[:, np.newaxis]).astype(np.float32)
    return cv2.remap(
        src=image.astype(np.float64),
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
    )


def _uniform_polar_radii(n_radial: int, radius: float) -> np.ndarray:
    """Compute the radii of a uniform polar array, in ``cartesian_to_polar``'s way.

    ``pypft.geometry.cartesian_to_polar`` places radial sample ``k`` at
    ``k * radius / n_radial``, ``k = 0 .. n_radial - 1``, with ``radius`` the largest
    circle inscribed in the image (``min(height, width) / 2``): the first sample is
    the center itself, and the last falls one step short of ``radius``.

    :param n_radial: The number of uniform radial samples.
    :type n_radial: int
    :param radius: The radius the uniform samples cover.
    :type radius: float
    :returns: The ``(n_radial,)`` uniform radii.
    :rtype: np.ndarray

    """
    return np.arange(n_radial) * (radius / n_radial)


def _validate_uniform_polar(
    values: np.ndarray,
    grid: PolarGrid,
    radius: float,
    *,
    caller: str,
    stacklevel: int,
) -> None:
    """Validate uniform polar data against the grid it is put on.

    Shared by every function that reads uniform polar data onto a grid
    (``resample_uniform_polar`` here, and ``pypft.ring``'s uniform-polar sampler),
    so both accept exactly the same input.

    :param values: A uniform polar array, ``(radial, angular)`` or a batch
        ``(radial, angular, batch)``.
    :type values: np.ndarray
    :param grid: The grid the data is put on.
    :type grid: PolarGrid
    :param radius: The radius the uniform samples cover.
    :type radius: float
    :param caller: The public function's name, for the error messages.
    :type caller: str
    :param stacklevel: The ``warnings.warn`` stack level that reaches the user's
        call: one for this helper, plus one per function between it and that call.
    :type stacklevel: int
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is neither a single sample nor a batch, has fewer
        than two radial samples, no spokes, or a non-finite element, or ``radius``
        is not strictly positive.
    :raises RadiusCoverageError: If ``grid.R`` exceeds ``radius``.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    NumpyValidator.type_is_ndarray(value=values)
    _value_is_polar_sample_or_batch(value=values)
    NumpyValidator.value_is_finite(value=values)
    _type_is_polar_grid(value=grid)
    FloatValidator.type_is_float(value=radius)
    FloatValidator.value_is_positive(value=radius)
    if grid.limit_kind is not LimitKind.SPACE_LIMITED:
        raise NotImplementedError(
            f"{caller} only supports LimitKind.SPACE_LIMITED grids"
        )
    n_radial, n_spokes = values.shape[PolarAxis.RADIAL], values.shape[PolarAxis.ANGULAR]
    if n_radial < _MIN_UNIFORM_RADIAL_SAMPLES:
        raise ValueError(
            f"values must have at least {_MIN_UNIFORM_RADIAL_SAMPLES} radial samples, "
            f"got {n_radial}"
        )
    if n_spokes == 0:
        raise ValueError("values must have at least one spoke, got 0")
    if grid.R > radius:
        raise RadiusCoverageError(
            f"grid.R={grid.R} exceeds the radius={radius} the uniform samples cover"
        )
    if n_spokes < grid.n_angular:
        warnings.warn(
            message=(
                f"values has {n_spokes} spokes, fewer than grid.n_angular="
                f"{grid.n_angular}: interpolating across spokes cannot add the "
                f"angular detail the data does not hold."
            ),
            category=AngularUpsamplingWarning,
            stacklevel=stacklevel,
        )


def _resample_matching_spokes(
    values: np.ndarray, grid: PolarGrid, radii: np.ndarray
) -> np.ndarray:
    """Interpolate each spoke along the radius alone, onto that spoke's grid radii.

    :param values: The uniform polar array, with exactly ``grid.n_angular`` spokes.
    :type values: np.ndarray
    :param grid: The grid to resample onto.
    :type grid: PolarGrid
    :param radii: The uniform radii of ``values``' radial axis.
    :type radii: np.ndarray
    :returns: The array resampled at ``grid.r``, in the ``(radial, angular[, batch])``
        layout.
    :rtype: np.ndarray

    """
    grid_radii = grid.r
    spokes = [
        # One cubic spline per spoke, fitted along the radius (axis 0) of that
        # spoke's own samples; a complex array is fitted as one complex spline,
        # so its real and imaginary parts are interpolated together.
        CubicSpline(x=radii, y=values[:, spoke, ...], axis=0)(grid_radii[spoke])
        for spoke in range(grid.n_angular)
    ]
    return np.stack(arrays=spokes, axis=PolarAxis.ANGULAR)


def _resample_other_spokes(
    values: np.ndarray, grid: PolarGrid, radii: np.ndarray
) -> np.ndarray:
    """Interpolate along the radius and, periodically, across the spokes.

    :param values: The uniform polar array, with any number of spokes.
    :type values: np.ndarray
    :param grid: The grid to resample onto.
    :type grid: PolarGrid
    :param radii: The uniform radii of ``values``' radial axis.
    :type radii: np.ndarray
    :returns: The array resampled at ``grid.r``/``grid.theta``, in the
        ``(radial, angular[, batch])`` layout.
    :rtype: np.ndarray

    """
    n_spokes = values.shape[PolarAxis.ANGULAR]
    # The data's own spokes, in the same centered order as every stored angular axis.
    angles = _angular_harmonics(n_spokes) * (2.0 * np.pi / n_spokes)
    # One radial spline through every data spoke (and batch element) at once.
    radial = CubicSpline(x=radii, y=values, axis=PolarAxis.RADIAL)
    # A periodic spline needs its first spoke repeated one full turn later.
    closed_angles = np.append(arr=angles, values=angles[0] + 2.0 * np.pi)

    grid_radii = grid.r
    spokes = []
    for spoke, angle in enumerate(grid.theta):
        # Every data spoke at this grid spoke's radii, then across the spokes at its
        # angle; a periodic spline wraps any angle onto the data's own turn.
        on_radii = radial(grid_radii[spoke])
        closed = np.concatenate(
            (on_radii, on_radii[:, :1, ...]), axis=PolarAxis.ANGULAR
        )
        angular = CubicSpline(
            x=closed_angles, y=closed, axis=PolarAxis.ANGULAR, bc_type="periodic"
        )
        spokes.append(angular(angle))
    return np.stack(arrays=spokes, axis=PolarAxis.ANGULAR)


def resample_uniform_polar(
    values: np.ndarray, grid: PolarGrid, *, radius: float
) -> np.ndarray:
    """Interpolate uniform polar data along each spoke onto ``grid``'s sample points.

    ``values`` is on a uniform polar grid: the same equally spaced radii on every
    spoke, in exactly ``pypft.geometry.cartesian_to_polar``'s convention -- radial
    sample ``k`` at ``k * radius / n``, ``k = 0 .. n - 1`` for ``n`` uniform radial
    samples -- and uniform spokes on a centered angular axis. For a
    ``cartesian_to_polar`` result, ``radius`` is ``min(height, width) / 2`` of the
    source image, so
    ``resample_uniform_polar(cartesian_to_polar(image, ...), grid, radius=...)`` is
    the documented path from a uniform polar image to ``forward_pft``. Sampling the
    source image directly with ``sample_cartesian`` avoids the second interpolation
    when the Cartesian image is at hand.

    When ``values`` has exactly ``grid.n_angular`` spokes, they are the grid's own
    spokes, and each one is interpolated along the radius alone with a cubic spline.
    Otherwise the spokes are interpolated as well, with a periodic cubic spline
    across the angle, which adds angular interpolation error on top of the radial
    one. Either way a complex array is interpolated as one complex spline, its real
    and imaginary parts together. Grid radii past the last uniform sample, by less
    than one uniform radial step since the grid stays inside ``grid.R <= radius``,
    are extrapolated from the outermost spline piece.

    Measured, interpolating along each spoke gives the same ``forward_pft`` result as
    sampling on the grid directly; see ``DESIGN_NOTES.md``, "Grid: the spatial row
    index is a spoke, and the transform identifies it with a harmonic."

    :param values: A uniform polar array, ``(radial, angular)`` or a batch
        ``(radial, angular, batch)``, real or complex, with at least two radial
        samples and a centered angular axis.
    :type values: np.ndarray
    :param grid: The space-limited grid to resample onto.
    :type grid: PolarGrid
    :param radius: The radius the uniform samples cover, in the same unit as
        ``grid.R``.
    :type radius: float
    :returns: ``values`` resampled at ``grid.r``/``grid.theta``, of shape
        ``(grid.n_radial, grid.n_angular[, batch])``: ``forward_pft``'s input shape.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is neither a single sample nor a batch, has fewer
        than two radial samples, no spokes, or a non-finite element, or ``radius``
        is not strictly positive.
    :raises RadiusCoverageError: If ``grid.R`` exceeds ``radius``: the grid would need
        samples the data does not have.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    # stacklevel=3: past the helper and this function, at the user's call.
    _validate_uniform_polar(
        values=values,
        grid=grid,
        radius=radius,
        caller="resample_uniform_polar",
        stacklevel=3,
    )
    n_radial, n_spokes = values.shape[PolarAxis.RADIAL], values.shape[PolarAxis.ANGULAR]
    radii = _uniform_polar_radii(n_radial=n_radial, radius=radius)
    if n_spokes == grid.n_angular:
        return _resample_matching_spokes(values=values, grid=grid, radii=radii)
    return _resample_other_spokes(values=values, grid=grid, radii=radii)


# ======================================================================================
# Adequacy and Nyquist guards -- warnings, never errors
# ======================================================================================

#: Coefficients of a log-log least-squares fit of the forward Gaussian oracle's
#: relative L2 error to ``(n_angular, n_radial)``:
#: ``log2(error) = intercept + a * log2(n_angular) + b * log2(n_radial)``. Fitted from
#: nine measured points, ``forward_pft`` of the centered ``exp(-r**2)`` on the grid
#: against ``pi * exp(-rho**2 / 4)``, spanning ``n_angular in (15, 32, 64)`` and
#: ``n_radial in (383, 767, 1535)`` at ``R = 40``; ``DESIGN_NOTES.md``, "Grid:
#: ``check_adequacy`` is fitted on the relative L2 error," lists them. The fit is
#: within a factor of 1.11 of every measured point (largest residual 0.152 in
#: ``log2``): each doubling of ``n_radial`` divides the error by about 1.6, and each
#: doubling of ``n_angular`` multiplies it by about 1.76.
_ADEQUACY_INTERCEPT_LOG2 = 0.506
_ADEQUACY_N_ANGULAR_COEFFICIENT = 0.818
_ADEQUACY_N_RADIAL_COEFFICIENT = -0.687

#: The predicted forward relative L2 error above which a grid is reported: Yao &
#: Baddour's own worked example, ``n_radial=382, n_angular=15, R=40``, measures 0.243
#: (predicted 0.219), so a grid is flagged when it is predicted to approximate the
#: continuous transform worse than that reference grid does.
_ADEQUACY_THRESHOLD = 0.25


def _predicted_forward_relative_l2(n_angular: int, n_radial: int) -> float:
    """Predict the forward PFT's relative L2 error for a given grid size.

    :param n_angular: The number of angular samples.
    :type n_angular: int
    :param n_radial: The number of radial samples.
    :type n_radial: int
    :returns: The predicted relative L2 error.
    :rtype: float

    """
    return float(
        2.0
        ** (
            _ADEQUACY_INTERCEPT_LOG2
            + _ADEQUACY_N_ANGULAR_COEFFICIENT * np.log2(n_angular)
            + _ADEQUACY_N_RADIAL_COEFFICIENT * np.log2(n_radial)
        )
    )


def check_adequacy(grid: PolarGrid) -> None:
    """Warn if ``grid``'s ``n_radial`` is likely too small for its ``n_angular``.

    Raising angular resolution alone destroys accuracy: doubling ``n_angular``
    without growing ``n_radial`` to match multiplies the error by about 1.76, to the
    point that a high-``n_angular``, modest-``n_radial`` grid's *maximum* dB error can
    turn positive -- meaning the reconstruction is worse than useless at its worst
    point. This uses the measured relation (see ``_ADEQUACY_INTERCEPT_LOG2``'s
    docstring) to predict the forward transform's relative L2 error and warns, with a
    suggested ``n_radial``, whenever that prediction is worse than the threshold
    (``_ADEQUACY_THRESHOLD``, the error of Yao & Baddour's own reference grid).

    The fit is on the relative L2 error of a *centered* Gaussian of width 1 at
    ``R = 40``, and the check sees only ``(n_angular, n_radial)``. An off-center
    function, or an ``R`` that is small relative to the object, can give a larger
    error than predicted, and that part of the error does not shrink with more radial
    samples: for a Gaussian centered at ``(2, -1)`` on the ``n_radial=382,
    n_angular=15, R=40`` grid, doubling ``n_radial`` leaves the relative L2 error at
    about 0.24. Silence here is necessary, not sufficient, for an accurate
    approximation. See ``DESIGN_NOTES.md``, "Grid: the spatial row index is a spoke,
    and the transform identifies it with a harmonic."

    :param grid: The grid to check.
    :type grid: PolarGrid
    :raises TypeError: If ``grid`` is not a ``PolarGrid``.

    """
    _type_is_polar_grid(value=grid)
    predicted = _predicted_forward_relative_l2(
        n_angular=grid.n_angular, n_radial=grid.n_radial
    )
    if predicted > _ADEQUACY_THRESHOLD:
        # Solve the fit for the n_radial whose predicted error meets the threshold.
        needed_log2_n_radial = (
            _ADEQUACY_INTERCEPT_LOG2
            + _ADEQUACY_N_ANGULAR_COEFFICIENT * np.log2(grid.n_angular)
            - np.log2(_ADEQUACY_THRESHOLD)
        ) / -_ADEQUACY_N_RADIAL_COEFFICIENT
        suggested_n_radial = int(np.ceil(2.0**needed_log2_n_radial))
        warnings.warn(
            message=(
                f"n_radial={grid.n_radial} is likely inadequate for "
                f"n_angular={grid.n_angular}: predicted forward relative L2 error is "
                f"{predicted:.2f}, worse than the {_ADEQUACY_THRESHOLD:.2f} target; "
                f"try n_radial >= {suggested_n_radial}."
            ),
            category=AdequacyWarning,
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
    _type_is_polar_grid(value=grid)
    FloatValidator.type_is_float(value=band_limit)
    FloatValidator.value_is_positive(value=band_limit)
    if grid.limit_kind is not LimitKind.SPACE_LIMITED:
        raise NotImplementedError(
            "check_nyquist_adequacy only supports LimitKind.SPACE_LIMITED grids"
        )

    # jn_zeros(0, N) returns the first N zeros of J_0; the N1-th one (Baddour's
    # notation) is the (n_radial + 1)-th zero, since n_radial itself is "N1 - 1".
    j_0_n1 = jn_zeros(n=0, nt=grid.n_radial + 1)[-1]
    required = band_limit * grid.R
    if j_0_n1 < required:
        suggested_n_radial = int(np.ceil(2.0 * band_limit * grid.R / np.pi))
        warnings.warn(
            message=(
                f"grid violates the DHT Nyquist condition: "
                f"j_(0,{grid.n_radial + 1})={j_0_n1:.3f} < "
                f"band_limit*R={required:.3f}; try n_radial >= "
                f"{suggested_n_radial} (Eq. 24's asymptotic estimate)."
            ),
            category=NyquistWarning,
            stacklevel=2,
        )
