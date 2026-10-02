"""Tests for ``pypft.ring``, the ring-consistent route.

The route samples every harmonic on its own true rings, applies the existing
per-harmonic DHT, and evaluates every harmonic at each spoke's own radii. These tests
pin that it approximates the continuous transform far better than the exact path where
the exact path's spoke-to-harmonic identification dominates (``DESIGN_NOTES.md``, "PFT:
the exact path and the ring-consistent route"), that its pieces agree with the exact
path where they must, that its inverse is approximate rather than exact, and its input
contract. Every tolerance is a measured value with a stated margin.
"""

from pathlib import Path

import numpy as np
import pytest

from pypft.dft import angular_dft, inverse_angular_dft
from pypft.dht import hankel_transform
from pypft.domains import PolarSpatialHarmonicSignal
from pypft.grid import (
    AngularUpsamplingWarning,
    LimitKind,
    PolarGrid,
    RadiusCoverageError,
    resample_uniform_polar,
    sample_cartesian,
)
from pypft.ring import (
    evaluate_frequency,
    evaluate_space,
    forward_pft_ring,
    inverse_pft_ring,
    sample_harmonics_cartesian,
    sample_harmonics_uniform_polar,
)
from pypft.transform import Direction, forward_pft, inverse_pft, scaled_hankel

# ======================================================================================
# Shared grids, test functions and helpers
# ======================================================================================

#: A small grid, fast to build and transform.
_GRID = PolarGrid(n_radial=64, n_angular=15, R=10.0)

#: Angles of the reference quadrature around every ring, as in the measurements of
#: ``DESIGN_NOTES.md``.
_REFERENCE_QUADRATURE = 4096


def _gaussian(
    r: np.ndarray, theta: np.ndarray, center: tuple[float, float], width: float = 1.0
) -> np.ndarray:
    """A Gaussian of the given width centered at ``center``, at polar points.

    :param r: The radii.
    :type r: np.ndarray
    :param theta: The angles, broadcastable against ``r``.
    :type theta: np.ndarray
    :param center: The ``(x, y)`` center.
    :type center: tuple[float, float]
    :param width: The width, ``exp(-|x - center|**2 / width**2)``.
    :type width: float
    :returns: The Gaussian at every point.
    :rtype: np.ndarray

    """
    x0, y0 = center
    distance = (r * np.cos(theta) - x0) ** 2 + (r * np.sin(theta) - y0) ** 2
    return np.exp(-distance / width**2)


def _gaussian_transform(
    grid: PolarGrid, center: tuple[float, float], width: float = 1.0
) -> np.ndarray:
    """The exact 2-D Fourier transform of ``_gaussian`` on ``grid``'s frequency points.

    :param grid: The grid whose ``rho``/``psi`` to evaluate at.
    :type grid: PolarGrid
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :param width: The width of the Gaussian.
    :type width: float
    :returns: The transform, ``(n_radial, n_angular)``.
    :rtype: np.ndarray

    """
    x0, y0 = center
    rho, psi = grid.rho.T, grid.psi[np.newaxis, :]
    phase = rho * (np.cos(psi) * x0 + np.sin(psi) * y0)
    return np.pi * width**2 * np.exp(-(rho**2) * width**2 / 4) * np.exp(-1j * phase)


def _uniform_samples(grid: PolarGrid, center: tuple[float, float]) -> np.ndarray:
    """Sample ``_gaussian`` on ``grid.n_radial`` uniform radii up to ``grid.R``.

    :param grid: The grid whose spokes, radial count and ``R`` to use.
    :type grid: PolarGrid
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: The ``(radial, angular)`` uniform polar samples.
    :rtype: np.ndarray

    """
    radii = np.arange(grid.n_radial) * (grid.R / grid.n_radial)
    return _gaussian(
        r=radii[:, np.newaxis], theta=grid.theta[np.newaxis, :], center=center
    )


def _quadrature_harmonics(grid: PolarGrid, center: tuple[float, float]) -> np.ndarray:
    """Every harmonic of ``_gaussian`` on its own rings, by a dense quadrature.

    :param grid: The grid whose harmonics and rings to use.
    :type grid: PolarGrid
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: The ``(radial, angular)`` spatial-harmonic array, on ``angular_dft``'s
        scale.
    :rtype: np.ndarray

    """
    angles = np.arange(_REFERENCE_QUADRATURE) * (2 * np.pi / _REFERENCE_QUADRATURE)
    result = np.empty((grid.n_radial, grid.n_angular), dtype=complex)
    for row, harmonic in enumerate(grid.harmonics):
        rings = _gaussian(
            r=grid.r[row][:, np.newaxis], theta=angles[np.newaxis, :], center=center
        )
        phases = np.exp(-1j * harmonic * angles)
        result[:, row] = rings @ phases * (grid.n_angular / _REFERENCE_QUADRATURE)
    return result


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    """The relative L2 error of ``values`` against ``reference``.

    :param values: The computed values.
    :type values: np.ndarray
    :param reference: The reference values.
    :type reference: np.ndarray
    :returns: ``||values - reference|| / ||reference||``.
    :rtype: float

    """
    return float(np.linalg.norm(values - reference) / np.linalg.norm(reference))


# ======================================================================================
# Accuracy against the continuous transform
# ======================================================================================

#: Measured relative L2 error of the route, from a 4096-angle quadrature on true
#: rings, for the Gaussian centered at (2, -1) on Yao & Baddour's grid: 1.95e-2
#: (angular truncation at 15 harmonics), against forward_pft's 0.244. Allowed with a
#: margin of ~1.3x.
_APPENDIX_RING_RTOL = 2.5e-2

#: forward_pft's own measured relative L2 error on the same case.
_APPENDIX_EXACT_PATH_ERROR = 0.24


def test_quadrature_route_reproduces_the_off_center_appendix_case() -> None:
    grid = PolarGrid(n_radial=382, n_angular=15, R=40.0)
    center = (2.0, -1.0)
    F_exact = _gaussian_transform(grid=grid, center=center)
    f_n = _quadrature_harmonics(grid=grid, center=center)
    F_n = scaled_hankel(
        values=f_n, grid=grid, direction=Direction.FORWARD, axis=0, angular_axis=1
    )
    ring_error = _relative_l2(evaluate_frequency(harmonics=F_n, grid=grid), F_exact)
    on_grid = _gaussian(r=grid.r.T, theta=grid.theta[np.newaxis, :], center=center)
    exact_path_error = _relative_l2(forward_pft(f=on_grid, grid=grid), F_exact)
    assert ring_error < _APPENDIX_RING_RTOL
    assert exact_path_error > _APPENDIX_EXACT_PATH_ERROR


#: Measured relative L2 error of forward_pft_ring from uniform polar data, using only
#: the data's own 15 spokes as the quadrature, on the same case: 2.91e-2 (the data's
#: spokes alias the harmonics beyond the grid's range). Margin ~1.4x.
_APPENDIX_UNIFORM_RTOL = 4e-2


def test_uniform_route_with_only_the_data_angles_on_the_appendix_case() -> None:
    grid = PolarGrid(n_radial=382, n_angular=15, R=40.0)
    center = (2.0, -1.0)
    F = forward_pft_ring(
        values=_uniform_samples(grid=grid, center=center), grid=grid, radius=grid.R
    )
    assert (
        _relative_l2(F, _gaussian_transform(grid=grid, center=center))
        < _APPENDIX_UNIFORM_RTOL
    )


#: Measured relative L2 error of forward_pft_ring from uniform polar data on a grid
#: whose 31 harmonics hold the whole function: 7.9e-6 (the radial cubic spline), where
#: forward_pft measures 0.51. Margin ~2.5x.
_RESOLVED_UNIFORM_RTOL = 2e-5


def test_uniform_route_is_accurate_when_the_harmonics_resolve_the_function() -> None:
    grid = PolarGrid(n_radial=128, n_angular=31, R=20.0)
    center = (1.0, -0.5)
    F = forward_pft_ring(
        values=_uniform_samples(grid=grid, center=center), grid=grid, radius=grid.R
    )
    assert (
        _relative_l2(F, _gaussian_transform(grid=grid, center=center))
        < _RESOLVED_UNIFORM_RTOL
    )


#: Measured relative L2 error of forward_pft_ring from a Cartesian image of a Gaussian
#: of width 6 pixels centered 8 pixels right of and 4 above the image center: 7.2e-3
#: (bilinear pixel interpolation), where forward_pft of sample_cartesian measures 1.12.
#: Margin ~2x.
_CARTESIAN_RTOL = 1.5e-2


def test_cartesian_route_follows_the_image_angle_convention() -> None:
    grid = PolarGrid(n_radial=64, n_angular=15, R=32.0)
    size, width, center = 72, 6.0, (8.0, -4.0)
    rows, columns = np.mgrid[0:size, 0:size].astype(float)
    # Image coordinates: x along the columns, y down the rows, from the center.
    x, y = columns - size / 2, rows - size / 2
    image = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / width**2)
    F_exact = _gaussian_transform(grid=grid, center=center, width=width)
    assert (
        _relative_l2(forward_pft_ring(values=image, grid=grid), F_exact)
        < _CARTESIAN_RTOL
    )
    exact_path = forward_pft(f=sample_cartesian(image=image, grid=grid).T, grid=grid)
    assert _relative_l2(exact_path, F_exact) > 1.0


#: Measured relative L2 error of evaluate_space, from the quadrature harmonics, against
#: the function on the grid's own spatial points: 2.4e-9, where the exact path's
#: inverse angular DFT of the same harmonics (the identification) measures 0.21.
#: Margin ~4x.
_EVALUATE_SPACE_RTOL = 1e-8


def test_evaluate_space_reaches_each_spokes_own_radii() -> None:
    grid = PolarGrid(n_radial=128, n_angular=31, R=20.0)
    center = (1.0, -0.5)
    f_n = _quadrature_harmonics(grid=grid, center=center)
    on_grid = _gaussian(r=grid.r.T, theta=grid.theta[np.newaxis, :], center=center)
    assert (
        _relative_l2(evaluate_space(harmonics=f_n, grid=grid), on_grid)
        < _EVALUATE_SPACE_RTOL
    )
    assert _relative_l2(inverse_angular_dft(X=f_n, axis=1), on_grid) > 0.1


# ======================================================================================
# Agreement with the exact path
# ======================================================================================


def test_uniform_sampler_has_the_dft_scale_and_sign() -> None:
    # A cubic polynomial in r, which the radial spline reproduces exactly, times an
    # angular profile with known harmonics: cos(theta) + 0.5 sin(2 theta).
    radii = np.arange(_GRID.n_radial) * (_GRID.R / _GRID.n_radial)
    theta = _GRID.theta

    def profile(r: np.ndarray) -> np.ndarray:
        return 1 + r - r**2 / 10 + r**3 / 100

    values = profile(radii)[:, np.newaxis] * (np.cos(theta) + 0.5 * np.sin(2 * theta))
    harmonics = sample_harmonics_uniform_polar(
        values=values, grid=_GRID, radius=_GRID.R
    )
    # Every coefficient c_n of exp(i n theta), times n_angular: angular_dft's scale.
    coefficients = {1: 0.5, -1: 0.5, 2: 0.25 / 1j, -2: -0.25 / 1j}
    for column, harmonic in enumerate(_GRID.harmonics):
        expected = (
            _GRID.n_angular
            * coefficients.get(int(harmonic), 0.0)
            * profile(_GRID.r[column])
        )
        np.testing.assert_allclose(harmonics[:, column], expected, rtol=0, atol=1e-12)
    # The same profile on rings of one radius matches angular_dft exactly.
    one = np.array([1.0])
    ring_values = profile(one) * (np.cos(theta) + 0.5 * np.sin(2 * theta))
    dft = angular_dft(x=ring_values, axis=0)
    for column, harmonic in enumerate(_GRID.harmonics):
        np.testing.assert_allclose(
            dft[column],
            _GRID.n_angular * coefficients.get(int(harmonic), 0.0) * profile(one),
            atol=1e-12,
        )


def test_circularly_symmetric_function_matches_the_exact_paths_harmonic_zero() -> None:
    zero = _GRID.n_angular // 2
    values = _uniform_samples(grid=_GRID, center=(0.0, 0.0))
    harmonics = sample_harmonics_uniform_polar(
        values=values, grid=_GRID, radius=_GRID.R
    )
    # True rings: a circularly symmetric function has harmonic 0 alone, to rounding,
    # unlike the exact path's on-grid samples.
    np.testing.assert_allclose(
        np.delete(arr=harmonics, obj=zero, axis=1), 0.0, atol=1e-13
    )
    F = forward_pft_ring(values=values, grid=_GRID, radius=_GRID.R)
    # Spoke 0 sits at harmonic 0's own radii, where the route is harmonic 0's DHT.
    harmonic_zero = harmonics[:, zero].real / _GRID.n_angular
    expected = 2 * np.pi * hankel_transform(f=harmonic_zero, n=0, R=_GRID.R)
    np.testing.assert_allclose(
        F[:, zero], expected, rtol=0, atol=1e-14 * np.abs(expected).max()
    )
    # The exact path gives the same spoke 0 for input that is constant on rings.
    exact_path = forward_pft(
        f=np.repeat(a=harmonic_zero[:, np.newaxis], repeats=_GRID.n_angular, axis=1),
        grid=_GRID,
    )
    np.testing.assert_allclose(
        F[:, zero], exact_path[:, zero], rtol=0, atol=1e-14 * np.abs(expected).max()
    )


def test_evaluate_frequency_composes_with_the_domain_chain() -> None:
    values = _uniform_samples(grid=_GRID, center=(1.0, -0.5))
    harmonics = sample_harmonics_uniform_polar(
        values=values, grid=_GRID, radius=_GRID.R
    )
    signal = PolarSpatialHarmonicSignal(values=harmonics, grid=_GRID)
    chained = evaluate_frequency(
        harmonics=signal.to_frequency_harmonic().values, grid=_GRID
    )
    np.testing.assert_allclose(
        chained,
        forward_pft_ring(values=values, grid=_GRID, radius=_GRID.R),
        rtol=0,
        atol=1e-13,
    )


# ======================================================================================
# The approximate inverse
# ======================================================================================

#: Measured relative L2 round-trip error, forward_pft_ring then inverse_pft_ring,
#: against the function on the grid's spatial points, for a Gaussian centered at
#: (1, -0.5) on a grid whose frequency-side central gap holds little of the transform:
#: 7.0e-4. Margin ~2.9x.
_ROUND_TRIP_RTOL = 2e-3

#: The round trip is approximate: it stays far above rounding error.
_ROUND_TRIP_FLOOR = 1e-6


def test_round_trip_is_approximate() -> None:
    grid = PolarGrid(n_radial=96, n_angular=15, R=20.0)
    center = (1.0, -0.5)
    F = forward_pft_ring(
        values=_uniform_samples(grid=grid, center=center), grid=grid, radius=grid.R
    )
    on_grid = _gaussian(r=grid.r.T, theta=grid.theta[np.newaxis, :], center=center)
    error = _relative_l2(inverse_pft_ring(F=F, grid=grid), on_grid)
    assert _ROUND_TRIP_FLOOR < error < _ROUND_TRIP_RTOL
    # The exact path's own round trip is exact, by contrast.
    assert (
        _relative_l2(
            inverse_pft(F=forward_pft(f=on_grid, grid=grid), grid=grid), on_grid
        )
        < 1e-12
    )


# ======================================================================================
# Batches and complex input
# ======================================================================================


def _batch() -> np.ndarray:
    """Three complex uniform polar samples on ``_GRID``, stacked on a trailing axis.

    :returns: The ``(radial, angular, 3)`` batch.
    :rtype: np.ndarray

    """
    centers = ((0.0, 0.0), (1.0, -0.5), (-2.0, 1.0))
    samples = [
        _uniform_samples(grid=_GRID, center=center) * np.exp(1j * index)
        for index, center in enumerate(centers)
    ]
    return np.stack(arrays=samples, axis=-1)


def test_batch_matches_one_sample_at_a_time() -> None:
    batch = _batch()
    forward = forward_pft_ring(values=batch, grid=_GRID, radius=_GRID.R)
    inverse = inverse_pft_ring(F=forward, grid=_GRID)
    space = evaluate_space(harmonics=batch, grid=_GRID)
    assert forward.shape == inverse.shape == space.shape == batch.shape
    for element in range(batch.shape[-1]):
        single = forward_pft_ring(
            values=batch[..., element], grid=_GRID, radius=_GRID.R
        )
        np.testing.assert_allclose(forward[..., element], single, rtol=0, atol=1e-13)
        np.testing.assert_allclose(
            inverse[..., element],
            inverse_pft_ring(F=single, grid=_GRID),
            rtol=0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            space[..., element],
            evaluate_space(harmonics=batch[..., element], grid=_GRID),
            rtol=0,
            atol=1e-13,
        )


def test_cartesian_batch_and_complex_image_are_linear() -> None:
    rows, columns = np.mgrid[0:24, 0:24].astype(float)
    real = np.exp(-((columns - 13) ** 2 + (rows - 11) ** 2) / 9)
    imaginary = np.exp(-((columns - 10) ** 2 + (rows - 12) ** 2) / 4)
    grid = PolarGrid(n_radial=16, n_angular=7, R=10.0)
    complex_image = sample_harmonics_cartesian(image=real + 1j * imaginary, grid=grid)
    parts = sample_harmonics_cartesian(
        image=real, grid=grid
    ) + 1j * sample_harmonics_cartesian(image=imaginary, grid=grid)
    np.testing.assert_allclose(complex_image, parts, rtol=0, atol=1e-12)
    batch = sample_harmonics_cartesian(
        image=np.stack(arrays=(real, imaginary), axis=-1), grid=grid
    )
    assert batch.shape == (16, 7, 2)
    np.testing.assert_allclose(
        batch[..., 1],
        sample_harmonics_cartesian(image=imaginary, grid=grid),
        atol=1e-12,
    )


# ======================================================================================
# Input contract
# ======================================================================================


def test_samplers_reject_band_limited_grids() -> None:
    grid = PolarGrid(
        n_radial=16, n_angular=7, R=10.0, limit_kind=LimitKind.BAND_LIMITED
    )
    with pytest.raises(NotImplementedError):
        sample_harmonics_cartesian(image=np.ones((24, 24)), grid=grid)
    with pytest.raises(NotImplementedError):
        sample_harmonics_uniform_polar(values=np.ones((16, 7)), grid=grid, radius=10.0)
    for function in (evaluate_frequency, evaluate_space):
        with pytest.raises(NotImplementedError):
            function(harmonics=np.ones((16, 7)), grid=grid)
    with pytest.raises(NotImplementedError):
        inverse_pft_ring(F=np.ones((16, 7)), grid=grid)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"image": [[1.0]]}, TypeError),
        ({"image": np.ones(24)}, ValueError),
        ({"image": np.ones((2, 24, 24, 1))}, ValueError),
        ({"image": np.full((24, 24), np.nan)}, ValueError),
        ({"image": np.ones((24, 24)), "n_quadrature": 0}, ValueError),
        ({"image": np.ones((24, 24)), "n_quadrature": 8.0}, TypeError),
        ({"image": np.ones((24, 24)), "grid": "grid"}, TypeError),
    ],
)
def test_sample_harmonics_cartesian_rejects_invalid_input(
    kwargs: dict, error: type
) -> None:
    arguments = {"grid": PolarGrid(n_radial=16, n_angular=7, R=10.0), **kwargs}
    with pytest.raises(error):
        sample_harmonics_cartesian(**arguments)


def test_sample_harmonics_uniform_polar_checks_radius_and_spokes() -> None:
    values = _uniform_samples(grid=_GRID, center=(0.0, 0.0))
    with pytest.raises(RadiusCoverageError):
        sample_harmonics_uniform_polar(values=values, grid=_GRID, radius=_GRID.R / 2)
    with pytest.raises(TypeError):
        sample_harmonics_uniform_polar(values=values, grid=_GRID, radius=10)
    with pytest.warns(AngularUpsamplingWarning):
        sample_harmonics_uniform_polar(
            values=values[:, ::2], grid=_GRID, radius=_GRID.R
        )


@pytest.mark.parametrize("function", [evaluate_frequency, evaluate_space])
@pytest.mark.parametrize(
    ("values", "batch_axis", "error"),
    [
        (np.ones((64, 14)), -1, ValueError),
        (np.ones((64, 15, 2, 2)), -1, ValueError),
        (np.ones((64, 15, 2)), 0, ValueError),
        (np.ones((64, 15)), 2, ValueError),
        ([[1.0]], -1, TypeError),
    ],
)
def test_evaluators_reject_invalid_input(
    function, values, batch_axis: int, error: type
) -> None:
    with pytest.raises(error):
        function(harmonics=values, grid=_GRID, batch_axis=batch_axis)


def test_inverse_pft_ring_rejects_non_finite_input() -> None:
    F = np.ones((64, 15), dtype=complex)
    F[3, 4] = np.inf
    with pytest.raises(ValueError):
        inverse_pft_ring(F=F, grid=_GRID)


def test_keyword_only_arguments() -> None:
    values = _uniform_samples(grid=_GRID, center=(0.0, 0.0))
    with pytest.raises(TypeError):
        sample_harmonics_uniform_polar(values, _GRID, _GRID.R)  # type: ignore[misc]
    with pytest.raises(TypeError):
        forward_pft_ring(values, _GRID, _GRID.R)  # type: ignore[misc]
    with pytest.raises(TypeError):
        sample_harmonics_cartesian(np.ones((24, 24)), _GRID, 64)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "batch_axis", "error"),
    [
        (np.ones((64, 14)), -1, ValueError),
        (np.ones((64, 15, 2, 2)), -1, ValueError),
        (np.ones((64, 15, 2)), 0, ValueError),
        (np.ones((64, 15)), 2, ValueError),
        ([[1.0]], -1, TypeError),
    ],
)
def test_inverse_pft_ring_rejects_invalid_input(
    values, batch_axis: int, error: type
) -> None:
    with pytest.raises(error):
        inverse_pft_ring(F=values, grid=_GRID, batch_axis=batch_axis)


def test_forward_pft_ring_checks_radius_type() -> None:
    values = _uniform_samples(grid=_GRID, center=(0.0, 0.0))
    with pytest.raises(TypeError):
        forward_pft_ring(values=values, grid=_GRID, radius=10)


def test_angular_upsampling_warning_points_at_the_callers_line() -> None:
    few_spokes = _uniform_samples(grid=_GRID, center=(0.0, 0.0))[:, ::2]
    calls = (
        lambda: sample_harmonics_uniform_polar(
            values=few_spokes, grid=_GRID, radius=_GRID.R
        ),
        lambda: forward_pft_ring(values=few_spokes, grid=_GRID, radius=_GRID.R),
        lambda: resample_uniform_polar(values=few_spokes, grid=_GRID, radius=_GRID.R),
    )
    for call in calls:
        with pytest.warns(AngularUpsamplingWarning) as record:
            call()
        # Path equality, since Windows reports the same file with either drive case.
        assert Path(record[0].filename) == Path(__file__)
