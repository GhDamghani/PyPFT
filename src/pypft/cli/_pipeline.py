"""The terminal interface's transform step: the choices it offers, and running them.

A thin layer over the public API, independent of the user interface: it builds the
``pypft.grid.PolarGrid`` the user describes, runs ``pypft.grid.check_adequacy`` and
``pypft.grid.check_nyquist_adequacy`` on it, and transforms a loaded array along the
chosen path, adding no numerics of its own. Every warning the API emits on the way is
recorded and returned as text, for the screen to show.

The two paths are the package's own:

- **Exact** (the default): ``pypft.forward_pft``/``pypft.inverse_pft`` on the grid,
  square and exactly invertible. Uniform polar input is first interpolated along each
  spoke onto the grid (``pypft.resample_uniform_polar``, or
  ``pypft.resample_uniform_polar_frequency`` for frequency data). Tracing records
  every domain the transform passes through (``pypft.forward_pft_traced``/
  ``pypft.inverse_pft_traced``).
- **Accurate**: the ring-consistent route, ``pypft.forward_pft_ring``/
  ``pypft.inverse_pft_ring``, which reads uniform polar data directly. Its forward
  direction needs uniform polar input, and its inverse is an approximation.

For uniform polar k-space, which records the radius it covers, ``suggested_grid``
proposes a grid that fits inside the data.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pypft.axes import PolarAxis
from pypft.domains import (
    BaseSignal,
    PolarFrequencyAngularSignal,
    PolarSpatialAngularSignal,
)
from pypft.grid import (
    LimitKind,
    PolarGrid,
    check_adequacy,
    check_nyquist_adequacy,
    resample_uniform_polar,
    resample_uniform_polar_frequency,
)
from pypft.ring import forward_pft_ring, inverse_pft_ring
from pypft.transform import Direction, forward_pft, inverse_pft
from pypft.utils.validators import (
    EnumValidator,
    FloatValidator,
    IntValidator,
    StrValidator,
)
from pypft.viz import forward_pft_traced, inverse_pft_traced

from ._io import LoadedInput, _type_is_loaded_input, record_warnings

# ======================================================================================
# Constants
# ======================================================================================

#: Shown whenever uniform polar input is interpolated onto the grid.
INTERPOLATION_NOTICE = (
    "Uniform polar input was interpolated along each spoke onto the grid: the "
    "result carries that interpolation's error."
)

#: The fraction of uniform k-space's spokes ``suggested_grid`` gives the grid: a
#: quarter, as in the experimental-data tutorial, keeps the transform short.
_SUGGESTED_SPOKE_DIVISOR = 4

#: Shown whenever the ring-consistent route's inverse runs.
APPROXIMATE_INVERSE_NOTICE = (
    "The ring-consistent route's inverse is an approximation, not an exact inverse "
    "of the forward transform."
)


# ======================================================================================
# Choices
# ======================================================================================


class InputGrid(Enum):
    """Where a file's samples sit."""

    #: On the transform's own grid, ``PolarGrid.r`` (or ``.rho``): fed as-is.
    ON_GRID = "on the transform's grid"
    #: On a uniform polar grid: the same equally spaced radii on every spoke.
    UNIFORM_POLAR = "uniform polar"


class Route(Enum):
    """Which of the package's two paths transforms the array."""

    EXACT = "exact"
    ACCURATE = "accurate (ring-consistent)"


@dataclass(frozen=True)
class GridSettings:
    """What the user says about a file's grid, which the file itself does not carry.

    :param input_grid: Where the file's samples sit.
    :type input_grid: InputGrid
    :param n_radial: The grid's radial sample count.
    :type n_radial: int
    :param n_angular: The grid's angular sample count.
    :type n_angular: int
    :param R: The grid's space limit (or band limit, for ``LimitKind.BAND_LIMITED``).
    :type R: float
    :param limit_kind: Whether ``R`` is a space limit or a band limit.
    :type limit_kind: pypft.grid.LimitKind
    :param radius: For uniform polar input, the radius its samples cover: in the unit
        of ``R`` for spatial data, in radians per unit of ``R`` for frequency data.
        ``None`` for input on the grid.
    :type radius: float | None
    :param band_limit: The band limit to check the grid's Nyquist condition against,
        or ``None`` to skip that check.
    :type band_limit: float | None
    :raises TypeError: If any field has the wrong type.
    :raises ValueError: If any number is not strictly positive, or ``radius`` is
        missing for uniform polar input or given for input on the grid.

    """

    input_grid: InputGrid
    n_radial: int
    n_angular: int
    R: float
    limit_kind: LimitKind = LimitKind.SPACE_LIMITED
    radius: float | None = None
    band_limit: float | None = None

    def __post_init__(self) -> None:
        """Validate every field once, right after construction."""
        EnumValidator.type_is_enum(value=self.input_grid)
        EnumValidator.value_is_enum_member(value=self.input_grid, enum_class=InputGrid)
        for count in (self.n_radial, self.n_angular):
            IntValidator.type_is_int(value=count)
            IntValidator.value_is_positive(value=count)
        EnumValidator.type_is_enum(value=self.limit_kind)
        EnumValidator.value_is_enum_member(value=self.limit_kind, enum_class=LimitKind)
        for number in (self.R, self.radius, self.band_limit):
            if number is not None:
                FloatValidator.type_is_float(value=number)
                FloatValidator.value_is_positive(value=number)
        uniform = self.input_grid is InputGrid.UNIFORM_POLAR
        if uniform and self.radius is None:
            raise ValueError("uniform polar input needs the radius its samples cover")
        if not uniform and self.radius is not None:
            raise ValueError("input on the transform's grid takes no radius")

    def grid(self) -> PolarGrid:
        """Build the grid these settings describe.

        :returns: The grid.
        :rtype: pypft.grid.PolarGrid

        """
        return PolarGrid(
            n_radial=self.n_radial,
            n_angular=self.n_angular,
            R=self.R,
            limit_kind=self.limit_kind,
        )


@dataclass(frozen=True)
class SuggestedGrid:
    """A grid size proposed for uniform polar k-space.

    :param n_radial: The grid's radial sample count.
    :type n_radial: int
    :param n_angular: The grid's angular sample count.
    :type n_angular: int
    :param R: The grid's space limit, in pixels.
    :type R: float

    """

    n_radial: int
    n_angular: int
    R: float


@dataclass(frozen=True)
class TransformResult:
    """A finished transform: the signals to show, and what the screen must say.

    :param signals: Every domain recorded, in pipeline order; the last is the result.
        A single signal unless the transform was traced.
    :type signals: tuple[pypft.domains.BaseSignal, ...]
    :param notices: Every notice and recorded warning, as text.
    :type notices: tuple[str, ...]

    """

    signals: tuple[BaseSignal, ...]
    notices: tuple[str, ...]


# ======================================================================================
# Parsing fields
# ======================================================================================


def parse_positive_float(text: str, *, name: str) -> float:
    """Read a strictly positive number from a text field.

    :param text: The field's text; surrounding spaces are ignored.
    :type text: str
    :param name: What the field holds, for the error message.
    :type name: str
    :returns: The number.
    :rtype: float
    :raises TypeError: If ``text`` or ``name`` is not a ``str``.
    :raises ValueError: If the text is not a finite, strictly positive number.

    """
    StrValidator.type_is_str(value=text)
    StrValidator.type_is_str(value=name)
    try:
        number = float(text.strip())
    except ValueError:
        raise ValueError(f"{name} must be a number, got {text!r}") from None
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive number, got {text!r}")
    return number


def parse_optional_positive_float(text: str, *, name: str) -> float | None:
    """Read a strictly positive number from a text field that may be left empty.

    :param text: The field's text; surrounding spaces are ignored.
    :type text: str
    :param name: What the field holds, for the error message.
    :type name: str
    :returns: The number, or ``None`` for an empty field.
    :rtype: float | None
    :raises TypeError: If ``text`` or ``name`` is not a ``str``.
    :raises ValueError: If the text is neither empty nor a finite, strictly positive
        number.

    """
    StrValidator.type_is_str(value=text)
    if not text.strip():
        return None
    return parse_positive_float(text=text, name=name)


def parse_positive_int(text: str, *, name: str) -> int:
    """Read a strictly positive integer from a text field.

    :param text: The field's text; surrounding spaces are ignored.
    :type text: str
    :param name: What the field holds, for the error message.
    :type name: str
    :returns: The integer.
    :rtype: int
    :raises TypeError: If ``text`` or ``name`` is not a ``str``.
    :raises ValueError: If the text is not a strictly positive integer.

    """
    StrValidator.type_is_str(value=text)
    StrValidator.type_is_str(value=name)
    try:
        number = int(text.strip())
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {text!r}") from None
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer, got {text!r}")
    return number


# ======================================================================================
# Running
# ======================================================================================


def suggested_grid(shape: tuple[int, ...], radius: float) -> SuggestedGrid:
    """Propose a space-limited grid for uniform polar k-space.

    - ``R`` is the half-width of the field of view the samples resolve: ``n * pi /
      radius`` for ``n`` samples per spoke spaced ``radius / n`` apart (for a readout
      of ``2 n`` samples, ``n`` pixels).
    - ``n_angular`` is a quarter of the data's spokes, at least one.
    - ``n_radial`` is the largest count whose frequencies stay inside the data,
      ``max(grid.rho) <= radius``, by the bound ``j_(m, k) < (k + m / 2 - 1 / 4) pi``
      on the Bessel zeros (``m >= 1 / 2``), with ``m`` the largest harmonic order; at
      least one.

    :param shape: The data's shape, ``(radial, angular[, batch])``.
    :type shape: tuple[int, ...]
    :param radius: The frequency radius the data covers, in radians per pixel.
    :type radius: float
    :returns: The proposed grid size and ``R``.
    :rtype: SuggestedGrid
    :raises TypeError: If ``radius`` is not a float.
    :raises ValueError: If ``radius`` is not positive, or ``shape`` has fewer than two
        positive sizes.

    """
    FloatValidator.type_is_float(value=radius)
    FloatValidator.value_is_positive(value=radius)
    if len(shape) < len(PolarAxis) or min(shape[: len(PolarAxis)]) <= 0:
        raise ValueError(f"shape must start with (radial, angular) sizes, got {shape}")
    n_data, n_spokes = shape[PolarAxis.RADIAL], shape[PolarAxis.ANGULAR]
    R = n_data * math.pi / radius
    n_angular = max(1, n_spokes // _SUGGESTED_SPOKE_DIVISOR)
    max_order = n_angular // 2
    # The largest zero j_(max_order, n_radial) must not pass radius * R.
    n_radial = math.floor(radius * R / math.pi - max_order / 2 + 1 / 4)
    return SuggestedGrid(n_radial=max(1, n_radial), n_angular=n_angular, R=R)


def check_grid(settings: GridSettings) -> tuple[PolarGrid, tuple[str, ...]]:
    """Build the grid and run the adequacy checks that apply to it.

    ``pypft.grid.check_adequacy`` always runs; ``pypft.grid.check_nyquist_adequacy``
    runs when the settings carry a band limit.

    :param settings: The grid's description.
    :type settings: GridSettings
    :returns: The grid, and every warning the checks emitted, as text.
    :rtype: tuple[pypft.grid.PolarGrid, tuple[str, ...]]
    :raises TypeError: If ``settings`` is not a ``GridSettings``.
    :raises NotImplementedError: If a band limit is given for a band-limited grid.

    """
    if not isinstance(settings, GridSettings):
        raise TypeError(
            f"settings must be {GridSettings.__name__}, got {type(settings).__name__}"
        )
    grid = settings.grid()

    def checks() -> None:
        check_adequacy(grid=grid)
        if settings.band_limit is not None:
            check_nyquist_adequacy(grid=grid, band_limit=settings.band_limit)

    _, messages = record_warnings(call=checks)
    return grid, tuple(messages)


def _validate_request(
    loaded: LoadedInput,
    settings: GridSettings,
    direction: Direction,
    route: Route,
    trace: bool,
) -> None:
    """Reject a combination of choices the package's API cannot run.

    :raises ValueError: If the combination is not supported; the message says why.

    """
    if direction is Direction.INVERSE and loaded.is_image:
        raise ValueError(
            "an image holds real spatial samples; the IPFT needs frequency-domain "
            "input, from a .npy, .npz or acquisition file"
        )
    if loaded.radius is not None:
        if direction is not Direction.INVERSE:
            raise ValueError(
                "the file holds radial k-space: transform it with the IPFT"
            )
        if settings.input_grid is not InputGrid.UNIFORM_POLAR:
            raise ValueError("the file holds uniform polar k-space, not grid samples")
    if trace and route is not Route.EXACT:
        raise ValueError("tracing every domain is available on the exact path only")
    if (
        route is Route.ACCURATE
        and direction is Direction.FORWARD
        and settings.input_grid is InputGrid.ON_GRID
    ):
        raise ValueError(
            "the ring-consistent route's PFT reads uniform polar input; choose the "
            "exact path for samples on the transform's grid"
        )
    if settings.input_grid is InputGrid.ON_GRID:
        expected = (settings.n_radial, settings.n_angular)
        found = loaded.values.shape[: len(PolarAxis)]
        if found != expected:
            raise ValueError(
                f"the array's samples {found} do not match the grid {expected}"
            )


def _exact(
    values: np.ndarray, grid: PolarGrid, *, direction: Direction, trace: bool
) -> tuple[BaseSignal, ...]:
    """Transform samples on the grid along the exact path.

    :returns: Every traced domain in order, or the result alone.
    :rtype: tuple[pypft.domains.BaseSignal, ...]

    """
    if trace:
        # No figures are requested, so the trace holds none to close.
        if direction is Direction.FORWARD:
            return forward_pft_traced(f=values, grid=grid).signals
        return inverse_pft_traced(F=values, grid=grid).signals
    if direction is Direction.FORWARD:
        return (
            PolarFrequencyAngularSignal(
                values=forward_pft(f=values, grid=grid), grid=grid
            ),
        )
    return (
        PolarSpatialAngularSignal(values=inverse_pft(F=values, grid=grid), grid=grid),
    )


def run_transform(
    loaded: LoadedInput,
    settings: GridSettings,
    *,
    direction: Direction,
    route: Route = Route.EXACT,
    trace: bool = False,
) -> TransformResult:
    """Transform a loaded array as the user chose.

    :param loaded: The array and what its file says about it.
    :type loaded: pypft.cli._io.LoadedInput
    :param settings: The grid's description.
    :type settings: GridSettings
    :param direction: ``Direction.FORWARD`` for the PFT (spatial input),
        ``Direction.INVERSE`` for the IPFT (frequency input).
    :type direction: pypft.transform.Direction
    :param route: The path to take.
    :type route: Route
    :param trace: Whether to record every domain (exact path only).
    :type trace: bool
    :returns: The signals to show and the notices to display.
    :rtype: TransformResult
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If the choices do not combine (see the module docstring), or
        the array does not fit the grid.
    :raises pypft.grid.RadiusCoverageError: If uniform polar input does not cover the
        grid.
    :raises NotImplementedError: If a step supports space-limited grids only.

    """
    _type_is_loaded_input(value=loaded)
    if not isinstance(settings, GridSettings):
        raise TypeError(
            f"settings must be {GridSettings.__name__}, got {type(settings).__name__}"
        )
    EnumValidator.type_is_enum(value=direction)
    EnumValidator.value_is_enum_member(value=direction, enum_class=Direction)
    EnumValidator.type_is_enum(value=route)
    EnumValidator.value_is_enum_member(value=route, enum_class=Route)
    if not isinstance(trace, bool):
        raise TypeError(f"trace must be bool, got {type(trace).__name__}")
    _validate_request(
        loaded=loaded, settings=settings, direction=direction, route=route, trace=trace
    )
    grid = settings.grid()
    notices: list[str] = []

    def transform() -> tuple[BaseSignal, ...]:
        values = loaded.values
        if route is Route.ACCURATE:
            # The route reads uniform polar data itself; on-grid input reaches here
            # only for the inverse, which takes grid samples with radius omitted.
            if direction is Direction.FORWARD:
                F = forward_pft_ring(values=values, grid=grid, radius=settings.radius)
                return (PolarFrequencyAngularSignal(values=F, grid=grid),)
            notices.append(APPROXIMATE_INVERSE_NOTICE)
            f = inverse_pft_ring(F=values, grid=grid, radius=settings.radius)
            return (PolarSpatialAngularSignal(values=f, grid=grid),)
        # GridSettings gives uniform polar input, and only it, a radius.
        if settings.radius is not None:
            resample = (
                resample_uniform_polar
                if direction is Direction.FORWARD
                else resample_uniform_polar_frequency
            )
            values = resample(values=values, grid=grid, radius=settings.radius)
            notices.append(INTERPOLATION_NOTICE)
        return _exact(values=values, grid=grid, direction=direction, trace=trace)

    signals, messages = record_warnings(call=transform)
    return TransformResult(signals=signals, notices=tuple(notices + messages))
