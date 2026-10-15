"""Tests for ``pypft.cli._pipeline``, the terminal interface's transform step.

The step adds no numerics, so every result is compared with the public function it
must have called, on the same input: ``forward_pft``/``inverse_pft``, the uniform polar
resamplers, the traced transforms and the ring-consistent route. The tests also pin
which combinations of choices are rejected, and that every warning is returned as text
instead of escaping (which ``filterwarnings = ["error"]`` would turn into a failure).
"""

import numpy as np
import pytest

from pypft.cli._io import LoadedInput
from pypft.cli._pipeline import (
    APPROXIMATE_INVERSE_NOTICE,
    INTERPOLATION_NOTICE,
    GridSettings,
    InputGrid,
    Route,
    check_grid,
    parse_optional_positive_float,
    parse_positive_float,
    parse_positive_int,
    run_transform,
    suggested_grid,
)
from pypft.domains import PolarDomain
from pypft.grid import (
    LimitKind,
    PolarGrid,
    RadiusCoverageError,
    resample_uniform_polar,
    resample_uniform_polar_frequency,
)
from pypft.ring import forward_pft_ring, inverse_pft_ring
from pypft.transform import Direction, forward_pft, inverse_pft
from pypft.viz import forward_pft_traced

#: A small grid, cheap for every route.
_N_RADIAL, _N_ANGULAR, _R = 24, 7, 4.0
_GRID = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)

#: The settings of input on ``_GRID``.
_ON_GRID = GridSettings(
    input_grid=InputGrid.ON_GRID, n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R
)


def _uniform(radius: float = _R) -> GridSettings:
    """The settings of uniform polar input onto ``_GRID``, covering ``radius``."""
    return GridSettings(
        input_grid=InputGrid.UNIFORM_POLAR,
        n_radial=_N_RADIAL,
        n_angular=_N_ANGULAR,
        R=_R,
        radius=radius,
    )


def _gaussian(n_radial: int, radius: float) -> np.ndarray:
    """A centered Gaussian on ``n_radial`` uniform radii of every spoke of _GRID."""
    r = np.arange(n_radial) * radius / n_radial
    return np.repeat(np.exp(-(r**2))[:, np.newaxis], repeats=_N_ANGULAR, axis=1)


def _loaded(
    values: np.ndarray, radius: float | None = None, is_image: bool = False
) -> LoadedInput:
    return LoadedInput(values=values, radius=radius, is_image=is_image)


#: Samples of a Gaussian on _GRID, in PyPFT's (radial, angular) layout.
_ON_GRID_VALUES = np.exp(-(_GRID.r.T**2))


# ======================================================================================
# Parsing and settings
# ======================================================================================


def test_parse_positive_numbers() -> None:
    assert parse_positive_float(text=" 2.5 ", name="R") == 2.5
    assert parse_positive_int(text="17", name="n") == 17
    assert parse_optional_positive_float(text="  ", name="b") is None
    assert parse_optional_positive_float(text="3", name="b") == 3.0
    for text in ("0", "-1", "nan", "inf", "x", ""):
        with pytest.raises(ValueError, match="R must be"):
            parse_positive_float(text=text, name="R")
    for text in ("0", "2.5", "x"):
        with pytest.raises(ValueError, match="n must be"):
            parse_positive_int(text=text, name="n")


def test_grid_settings_require_a_radius_for_uniform_input_only() -> None:
    with pytest.raises(ValueError, match="needs the radius"):
        GridSettings(input_grid=InputGrid.UNIFORM_POLAR, n_radial=4, n_angular=3, R=1.0)
    with pytest.raises(ValueError, match="takes no radius"):
        GridSettings(
            input_grid=InputGrid.ON_GRID, n_radial=4, n_angular=3, R=1.0, radius=2.0
        )
    with pytest.raises(ValueError, match="positive"):
        GridSettings(input_grid=InputGrid.ON_GRID, n_radial=4, n_angular=3, R=-1.0)
    with pytest.raises(TypeError, match="float"):
        GridSettings(input_grid=InputGrid.ON_GRID, n_radial=4, n_angular=3, R=1)
    assert _ON_GRID.grid() == _GRID


def test_suggested_grid_for_the_md_cnn_excerpt() -> None:
    # 208 samples per spoke up to pi rad/pixel: a 416-pixel field of view.
    suggestion = suggested_grid(shape=(208, 392, 4), radius=np.pi)
    assert (suggestion.n_radial, suggestion.n_angular) == (183, 98)
    assert suggestion.R == pytest.approx(208.0)


@pytest.mark.parametrize(
    ("shape", "radius"), [((40, 24), 1.0), ((30, 9), 2.5), ((12, 3), 0.7)]
)
def test_suggested_grid_stays_inside_the_data(
    shape: tuple[int, int], radius: float
) -> None:
    suggestion = suggested_grid(shape=shape, radius=radius)
    grid = PolarGrid(
        n_radial=suggestion.n_radial,
        n_angular=suggestion.n_angular,
        R=suggestion.R,
    )
    assert np.max(a=grid.rho) <= radius
    # The data covers the grid, so the exact path's resampler accepts it.
    values = np.ones(shape=shape, dtype=complex)
    resample_uniform_polar_frequency(values=values, grid=grid, radius=radius)


def test_suggested_grid_validates_its_input() -> None:
    with pytest.raises(TypeError, match="float"):
        suggested_grid(shape=(8, 4), radius=1)
    with pytest.raises(ValueError, match="positive"):
        suggested_grid(shape=(8, 4), radius=-1.0)
    with pytest.raises(ValueError, match="radial, angular"):
        suggested_grid(shape=(8,), radius=1.0)


def test_check_grid_returns_warnings_as_text() -> None:
    # The maze fixture's grid fails check_adequacy; a band limit adds the Nyquist check.
    settings = GridSettings(
        input_grid=InputGrid.ON_GRID,
        n_radial=576,
        n_angular=48,
        R=1.0,
        band_limit=1e4,
    )
    grid, notices = check_grid(settings=settings)
    assert grid == settings.grid()
    assert [notice.split(":")[0] for notice in notices] == [
        "AdequacyWarning",
        "NyquistWarning",
    ]


def test_check_grid_passes_an_adequate_grid_silently() -> None:
    settings = GridSettings(
        input_grid=InputGrid.ON_GRID, n_radial=383, n_angular=15, R=40.0
    )
    assert check_grid(settings=settings)[1] == ()


def test_check_grid_reports_an_unsupported_band_limit_check() -> None:
    settings = GridSettings(
        input_grid=InputGrid.ON_GRID,
        n_radial=8,
        n_angular=3,
        R=1.0,
        limit_kind=LimitKind.BAND_LIMITED,
        band_limit=1.0,
    )
    with pytest.raises(NotImplementedError):
        check_grid(settings=settings)


# ======================================================================================
# The exact path
# ======================================================================================


def test_exact_path_on_the_grid_is_forward_pft_and_inverse_pft() -> None:
    forward = run_transform(
        loaded=_loaded(values=_ON_GRID_VALUES),
        settings=_ON_GRID,
        direction=Direction.FORWARD,
    )
    (signal,) = forward.signals
    assert signal.domain is PolarDomain.FREQUENCY_ANGULAR
    np.testing.assert_array_equal(
        signal.values, forward_pft(f=_ON_GRID_VALUES, grid=_GRID)
    )
    inverse = run_transform(
        loaded=_loaded(values=signal.values),
        settings=_ON_GRID,
        direction=Direction.INVERSE,
    )
    assert inverse.signals[-1].domain is PolarDomain.SPATIAL_ANGULAR
    np.testing.assert_array_equal(
        inverse.signals[-1].values, inverse_pft(F=signal.values, grid=_GRID)
    )


def test_uniform_input_goes_through_resample_uniform_polar() -> None:
    values = _gaussian(n_radial=40, radius=_R)
    result = run_transform(
        loaded=_loaded(values=values), settings=_uniform(), direction=Direction.FORWARD
    )
    expected = forward_pft(
        f=resample_uniform_polar(values=values, grid=_GRID, radius=_R), grid=_GRID
    )
    np.testing.assert_array_equal(result.signals[-1].values, expected)
    assert INTERPOLATION_NOTICE in result.notices


def test_uniform_frequency_input_goes_through_the_frequency_resampler() -> None:
    radius = 1.5 * float(np.max(a=_GRID.rho))
    values = _gaussian(n_radial=40, radius=radius) + 0j
    result = run_transform(
        loaded=_loaded(values=values, radius=radius),
        settings=_uniform(radius=radius),
        direction=Direction.INVERSE,
    )
    expected = inverse_pft(
        F=resample_uniform_polar_frequency(values=values, grid=_GRID, radius=radius),
        grid=_GRID,
    )
    np.testing.assert_array_equal(result.signals[-1].values, expected)


def test_tracing_records_every_domain() -> None:
    result = run_transform(
        loaded=_loaded(values=_ON_GRID_VALUES),
        settings=_ON_GRID,
        direction=Direction.FORWARD,
        trace=True,
    )
    trace = forward_pft_traced(f=_ON_GRID_VALUES, grid=_GRID)
    assert [s.domain for s in result.signals] == [s.domain for s in trace.signals]
    for found, expected in zip(result.signals, trace.signals):
        np.testing.assert_array_equal(found.values, expected.values)


def test_a_batch_is_transformed_whole() -> None:
    batch = np.stack(arrays=[_ON_GRID_VALUES, 2 * _ON_GRID_VALUES], axis=-1)
    result = run_transform(
        loaded=_loaded(values=batch), settings=_ON_GRID, direction=Direction.FORWARD
    )
    np.testing.assert_array_equal(
        result.signals[-1].values, forward_pft(f=batch, grid=_GRID)
    )


def test_uniform_input_must_cover_the_grid() -> None:
    with pytest.raises(RadiusCoverageError):
        run_transform(
            loaded=_loaded(values=_gaussian(n_radial=40, radius=1.0)),
            settings=_uniform(radius=_R / 2),
            direction=Direction.FORWARD,
        )


# ======================================================================================
# The ring-consistent route
# ======================================================================================


def test_accurate_forward_is_forward_pft_ring_on_uniform_input() -> None:
    values = _gaussian(n_radial=40, radius=_R)
    result = run_transform(
        loaded=_loaded(values=values),
        settings=_uniform(),
        direction=Direction.FORWARD,
        route=Route.ACCURATE,
    )
    np.testing.assert_array_equal(
        result.signals[-1].values,
        forward_pft_ring(values=values, grid=_GRID, radius=_R),
    )
    assert INTERPOLATION_NOTICE not in result.notices


def test_accurate_inverse_is_inverse_pft_ring_and_says_it_approximates() -> None:
    F = forward_pft(f=_ON_GRID_VALUES, grid=_GRID)
    result = run_transform(
        loaded=_loaded(values=F),
        settings=_ON_GRID,
        direction=Direction.INVERSE,
        route=Route.ACCURATE,
    )
    np.testing.assert_array_equal(
        result.signals[-1].values, inverse_pft_ring(F=F, grid=_GRID)
    )
    assert APPROXIMATE_INVERSE_NOTICE in result.notices


# ======================================================================================
# Rejected combinations
# ======================================================================================


@pytest.mark.parametrize(
    ("loaded", "settings", "direction", "route", "trace", "match"),
    [
        (
            _loaded(values=_ON_GRID_VALUES, is_image=True),
            _ON_GRID,
            Direction.INVERSE,
            Route.EXACT,
            False,
            "image",
        ),
        (
            _loaded(values=_gaussian(n_radial=40, radius=_R), radius=_R),
            _uniform(),
            Direction.FORWARD,
            Route.EXACT,
            False,
            "IPFT",
        ),
        (
            _loaded(values=_ON_GRID_VALUES, radius=_R),
            _ON_GRID,
            Direction.INVERSE,
            Route.EXACT,
            False,
            "not grid samples",
        ),
        (
            _loaded(values=_ON_GRID_VALUES),
            _ON_GRID,
            Direction.FORWARD,
            Route.ACCURATE,
            True,
            "exact path only",
        ),
        (
            _loaded(values=_ON_GRID_VALUES),
            _ON_GRID,
            Direction.FORWARD,
            Route.ACCURATE,
            False,
            "uniform polar input",
        ),
        (
            _loaded(values=_ON_GRID_VALUES[:-1]),
            _ON_GRID,
            Direction.FORWARD,
            Route.EXACT,
            False,
            "do not match the grid",
        ),
    ],
)
def test_unsupported_combinations_are_rejected(
    loaded: LoadedInput,
    settings: GridSettings,
    direction: Direction,
    route: Route,
    trace: bool,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_transform(
            loaded=loaded,
            settings=settings,
            direction=direction,
            route=route,
            trace=trace,
        )


def test_run_transform_validates_argument_types() -> None:
    with pytest.raises(TypeError, match="LoadedInput"):
        run_transform(
            loaded=_ON_GRID_VALUES,  # type: ignore[arg-type]
            settings=_ON_GRID,
            direction=Direction.FORWARD,
        )
    with pytest.raises(TypeError, match="GridSettings"):
        run_transform(
            loaded=_loaded(values=_ON_GRID_VALUES),
            settings=_GRID,  # type: ignore[arg-type]
            direction=Direction.FORWARD,
        )
    with pytest.raises(TypeError, match="Enum"):
        run_transform(
            loaded=_loaded(values=_ON_GRID_VALUES),
            settings=_ON_GRID,
            direction="forward",  # type: ignore[arg-type]
        )
