"""Tests for ``pypft.grid.resample_uniform_polar``, the uniform-polar-to-grid bridge.

The resampler interpolates uniform polar data, in ``pypft.geometry.cartesian_to_polar``'s
radius convention, along each spoke onto ``PolarGrid.r``. These tests pin that it
reproduces exactly what a cubic spline can represent, that it gives the same
``forward_pft`` result as sampling on the grid (``DESIGN_NOTES.md``, "Grid: the spatial
row index is a spoke, and the transform identifies it with a harmonic"), that the
``cartesian_to_polar`` route agrees with ``sample_cartesian``, and its input contract.
Every tolerance is a measured value with a stated margin.
"""

import numpy as np
import pytest

from pypft.dft import harmonics
from pypft.geometry import cartesian_to_polar
from pypft.grid import (
    AngularUpsamplingWarning,
    LimitKind,
    PolarGrid,
    RadiusCoverageError,
    resample_uniform_polar,
    sample_cartesian,
)
from pypft.transform import forward_pft

# ======================================================================================
# Shared grid and test functions
# ======================================================================================

#: A small grid, fast to build and transform.
_GRID = PolarGrid(n_radial=64, n_angular=15, R=10.0)

#: The radius the uniform samples cover in most tests: exactly the grid's ``R``.
_RADIUS = 10.0


def _uniform_radii(n_radial: int, radius: float) -> np.ndarray:
    """Uniform radii in ``cartesian_to_polar``'s convention: ``k * radius / n``.

    :param n_radial: The number of uniform radial samples.
    :type n_radial: int
    :param radius: The radius the samples cover.
    :type radius: float
    :returns: The uniform radii.
    :rtype: np.ndarray

    """
    return np.arange(n_radial) * (radius / n_radial)


def _spoke_angles(n_angular: int) -> np.ndarray:
    """The centered spoke angles of an angular axis with ``n_angular`` samples.

    :param n_angular: The number of spokes.
    :type n_angular: int
    :returns: The angles, in radians.
    :rtype: np.ndarray

    """
    return harmonics(n_angular=n_angular) * (2 * np.pi / n_angular)


def _cubic(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """A complex function that is a cubic polynomial in ``r`` on every spoke.

    :param r: The radii.
    :type r: np.ndarray
    :param theta: The angles, broadcastable against ``r``.
    :type theta: np.ndarray
    :returns: The function at every point.
    :rtype: np.ndarray

    """
    real = (1 + np.cos(theta)) * (0.5 * r**3 - 2 * r**2 + r - 3)
    imaginary = np.sin(2 * theta) * r**2
    return real + 1j * imaginary


def _gaussian(
    r: np.ndarray, theta: np.ndarray, center: tuple[float, float]
) -> np.ndarray:
    """Evaluate ``exp(-|x - center|**2)`` at polar points.

    :param r: The radii.
    :type r: np.ndarray
    :param theta: The angles, broadcastable against ``r``.
    :type theta: np.ndarray
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: The Gaussian at every point.
    :rtype: np.ndarray

    """
    x0, y0 = center
    return np.exp(-((r * np.cos(theta) - x0) ** 2 + (r * np.sin(theta) - y0) ** 2))


def _smooth(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """A smooth function with a little angular structure, for spoke interpolation.

    :param r: The radii.
    :type r: np.ndarray
    :param theta: The angles, broadcastable against ``r``.
    :type theta: np.ndarray
    :returns: The function at every point.
    :rtype: np.ndarray

    """
    return np.exp(-(r**2) / 20) * (1 + 0.3 * np.cos(theta) + 0.1 * np.sin(theta))


def _relative_l2(approximation: np.ndarray, exact: np.ndarray) -> float:
    """The relative L2 error of ``approximation`` against ``exact``.

    :param approximation: The computed array.
    :type approximation: np.ndarray
    :param exact: The reference array.
    :type exact: np.ndarray
    :returns: ``||approximation - exact|| / ||exact||``.
    :rtype: float

    """
    return float(np.linalg.norm(approximation - exact) / np.linalg.norm(exact))


# ======================================================================================
# Exact reproduction
# ======================================================================================

#: Measured largest error relative to the function's largest magnitude: 2.0e-16 from
#: as many uniform radii as grid radii, 1.4e-15 from 30.
_CUBIC_RTOL = 1e-12


@pytest.mark.parametrize(argnames="n_uniform", argvalues=[64, 30])
def test_a_cubic_in_r_on_every_spoke_is_reproduced_exactly(n_uniform: int) -> None:
    """A cubic spline reproduces a cubic polynomial, extrapolated piece included."""
    uniform = _cubic(
        r=_uniform_radii(n_radial=n_uniform, radius=_RADIUS)[:, np.newaxis],
        theta=_GRID.theta[np.newaxis, :],
    )
    resampled = resample_uniform_polar(values=uniform, grid=_GRID, radius=_RADIUS)
    exact = _cubic(r=_GRID.r.T, theta=_GRID.theta[np.newaxis, :])
    assert resampled.shape == (_GRID.n_radial, _GRID.n_angular)
    assert np.max(np.abs(resampled - exact)) / np.max(np.abs(exact)) < _CUBIC_RTOL


def test_real_input_stays_real_and_complex_parts_resample_together() -> None:
    """Real input gives a real result; a complex one equals its parts resampled."""
    uniform = _cubic(
        r=_uniform_radii(n_radial=40, radius=_RADIUS)[:, np.newaxis],
        theta=_GRID.theta[np.newaxis, :],
    )
    real = resample_uniform_polar(values=uniform.real, grid=_GRID, radius=_RADIUS)
    imaginary = resample_uniform_polar(values=uniform.imag, grid=_GRID, radius=_RADIUS)
    combined = resample_uniform_polar(values=uniform, grid=_GRID, radius=_RADIUS)
    assert not np.iscomplexobj(real)
    np.testing.assert_allclose(actual=combined, desired=real + 1j * imaginary)


def test_a_batch_resamples_each_element_like_a_single_sample() -> None:
    """A trailing batch axis rides along: every element matches its own call."""
    radii = _uniform_radii(n_radial=48, radius=_RADIUS)[:, np.newaxis]
    single = _smooth(r=radii, theta=_GRID.theta[np.newaxis, :])
    batch = np.stack(arrays=[single, 2 * single, single**2], axis=-1)
    resampled = resample_uniform_polar(values=batch, grid=_GRID, radius=_RADIUS)
    assert resampled.shape == (_GRID.n_radial, _GRID.n_angular, 3)
    for element in range(batch.shape[-1]):
        np.testing.assert_allclose(
            actual=resampled[..., element],
            desired=resample_uniform_polar(
                values=batch[..., element], grid=_GRID, radius=_RADIUS
            ),
        )


# ======================================================================================
# Equivalence to sampling on the grid
# ======================================================================================

#: Measured relative L2 difference between ``forward_pft`` of the resampled uniform
#: samples and ``forward_pft`` of the on-grid samples, from 64 uniform radii: 1.1e-5
#: centered, 4.2e-6 off-center. Allowed here with a margin of ~10x over the larger.
_ON_GRID_RTOL = 1e-4


@pytest.mark.parametrize(argnames="center", argvalues=[(0.0, 0.0), (2.0, -1.0)])
def test_resampled_uniform_samples_transform_like_on_grid_samples(
    center: tuple[float, float],
) -> None:
    """Uniform samples, resampled onto ``grid.r``, transform as on-grid samples do."""
    uniform = _gaussian(
        r=_uniform_radii(n_radial=_GRID.n_radial, radius=_RADIUS)[:, np.newaxis],
        theta=_GRID.theta[np.newaxis, :],
        center=center,
    )
    from_uniform = forward_pft(
        f=resample_uniform_polar(values=uniform, grid=_GRID, radius=_RADIUS) + 0j,
        grid=_GRID,
    )
    on_grid = forward_pft(
        f=_gaussian(r=_GRID.r.T, theta=_GRID.theta[np.newaxis, :], center=center) + 0j,
        grid=_GRID,
    )
    assert _relative_l2(approximation=from_uniform, exact=on_grid) < _ON_GRID_RTOL


#: Measured relative L2 difference between ``forward_pft`` through
#: ``cartesian_to_polar`` and ``resample_uniform_polar`` and ``forward_pft`` through
#: ``sample_cartesian``, on the image below: 2.7e-3. ``cartesian_to_polar`` samples
#: the image at the nearest pixel, which dominates this difference. Allowed here with
#: a margin of ~3.6x.
_CARTESIAN_ROUTE_RTOL = 1e-2


def test_the_cartesian_to_polar_route_agrees_with_sample_cartesian() -> None:
    """``cartesian_to_polar`` then resampling matches ``sample_cartesian``."""
    size = 256
    rows, cols = np.mgrid[0:size, 0:size]
    x, y = cols - size / 2, rows - size / 2
    image = np.exp(-((x - 30) ** 2 + y**2) / (2 * 20**2))
    image += 0.6 * np.exp(-(x**2 + (y + 40) ** 2) / (2 * 25**2))
    grid = PolarGrid(n_radial=96, n_angular=15, R=100.0)

    # The image's own inscribed radius is cartesian_to_polar's radius.
    uniform = cartesian_to_polar(image=image, n_radial=128, n_angular=grid.n_angular)
    through_uniform = forward_pft(
        f=resample_uniform_polar(values=uniform, grid=grid, radius=size / 2) + 0j,
        grid=grid,
    )
    direct = forward_pft(f=sample_cartesian(image=image, grid=grid).T + 0j, grid=grid)
    assert _relative_l2(approximation=through_uniform, exact=direct) < (
        _CARTESIAN_ROUTE_RTOL
    )


# ======================================================================================
# A different number of spokes
# ======================================================================================

#: Measured largest error of the smooth function resampled from 31 spokes onto the
#: grid's 15: 1.3e-6. Allowed here with a margin of ~7x.
_MORE_SPOKES_ATOL = 1e-5


def test_more_spokes_are_interpolated_across_the_angle_without_a_warning() -> None:
    """Data with more spokes than the grid is interpolated periodically, silently."""
    n_spokes = 31
    uniform = _smooth(
        r=_uniform_radii(n_radial=_GRID.n_radial, radius=_RADIUS)[:, np.newaxis],
        theta=_spoke_angles(n_angular=n_spokes)[np.newaxis, :],
    )
    # filterwarnings = ["error"] turns any warning here into a failure.
    resampled = resample_uniform_polar(values=uniform, grid=_GRID, radius=_RADIUS)
    exact = _smooth(r=_GRID.r.T, theta=_GRID.theta[np.newaxis, :])
    assert resampled.shape == (_GRID.n_radial, _GRID.n_angular)
    np.testing.assert_allclose(
        actual=resampled, desired=exact, rtol=0.0, atol=_MORE_SPOKES_ATOL
    )


@pytest.mark.parametrize(argnames="n_spokes", argvalues=[7, 1])
def test_fewer_spokes_than_the_grid_warn(n_spokes: int) -> None:
    """Interpolating across fewer spokes than the grid has warns, then resamples."""
    uniform = _smooth(
        r=_uniform_radii(n_radial=_GRID.n_radial, radius=_RADIUS)[:, np.newaxis],
        theta=_spoke_angles(n_angular=n_spokes)[np.newaxis, :],
    )
    with pytest.warns(AngularUpsamplingWarning, match="spokes"):
        resampled = resample_uniform_polar(values=uniform, grid=_GRID, radius=_RADIUS)
    assert resampled.shape == (_GRID.n_radial, _GRID.n_angular)
    assert np.all(np.isfinite(resampled))


# ======================================================================================
# Input contract
# ======================================================================================


def _valid_values() -> np.ndarray:
    """A valid uniform polar array for ``_GRID``."""
    return np.ones(shape=(32, _GRID.n_angular))


@pytest.mark.parametrize(
    argnames="shape", argvalues=[(32,), (32, _GRID.n_angular, 2, 2)]
)
def test_rejects_values_that_are_not_2d_or_3d(shape: tuple[int, ...]) -> None:
    """Only a single sample or a batch with one trailing batch axis is accepted."""
    with pytest.raises(ValueError):
        resample_uniform_polar(values=np.ones(shape=shape), grid=_GRID, radius=_RADIUS)


def test_rejects_a_single_radial_sample() -> None:
    """A cubic spline needs at least two radial samples."""
    with pytest.raises(ValueError, match="radial samples"):
        resample_uniform_polar(
            values=np.ones(shape=(1, _GRID.n_angular)), grid=_GRID, radius=_RADIUS
        )


def test_rejects_values_without_spokes() -> None:
    """An empty angular axis has nothing to interpolate."""
    with pytest.raises(ValueError, match="spoke"):
        resample_uniform_polar(
            values=np.ones(shape=(16, 0)), grid=_GRID, radius=_RADIUS
        )


def test_rejects_non_finite_values() -> None:
    """A ``NaN`` sample would spread through its whole spline."""
    values = _valid_values()
    values[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resample_uniform_polar(values=values, grid=_GRID, radius=_RADIUS)


@pytest.mark.parametrize(
    argnames=("kwargs", "error"),
    argvalues=[
        ({"values": [[1.0, 2.0]]}, TypeError),
        ({"grid": (64, 15, 10.0)}, TypeError),
        ({"radius": 10}, TypeError),
        ({"radius": 0.0}, ValueError),
        ({"radius": -1.0}, ValueError),
    ],
)
def test_rejects_invalid_arguments(kwargs: dict, error: type) -> None:
    """Every argument is type- and value-validated."""
    arguments = {"values": _valid_values(), "grid": _GRID, "radius": _RADIUS}
    arguments.update(kwargs)
    with pytest.raises(error):
        resample_uniform_polar(**arguments)


def test_rejects_a_grid_reaching_past_the_data() -> None:
    """``grid.R > radius`` would need samples the data does not have."""
    with pytest.raises(RadiusCoverageError, match="exceeds"):
        resample_uniform_polar(values=_valid_values(), grid=_GRID, radius=9.0)
    assert issubclass(RadiusCoverageError, ValueError)


def test_accepts_data_reaching_past_the_grid() -> None:
    """Data covering more than ``grid.R`` is fine: the grid uses only part of it."""
    resampled = resample_uniform_polar(values=_valid_values(), grid=_GRID, radius=20.0)
    np.testing.assert_allclose(actual=resampled, desired=1.0)


def test_rejects_band_limited_grids() -> None:
    """``R`` is a band limit there, which no data radius can be compared with."""
    grid = PolarGrid(
        n_radial=64, n_angular=15, R=10.0, limit_kind=LimitKind.BAND_LIMITED
    )
    with pytest.raises(NotImplementedError):
        resample_uniform_polar(values=_valid_values(), grid=grid, radius=_RADIUS)


def test_radius_is_keyword_only() -> None:
    """``radius`` cannot be passed positionally."""
    with pytest.raises(TypeError):
        resample_uniform_polar(_valid_values(), _GRID, _RADIUS)  # type: ignore[misc]


def test_does_not_modify_its_input() -> None:
    """The input array is left untouched."""
    values = _smooth(
        r=_uniform_radii(n_radial=32, radius=_RADIUS)[:, np.newaxis],
        theta=_GRID.theta[np.newaxis, :],
    )
    original = values.copy()
    resample_uniform_polar(values=values, grid=_GRID, radius=_RADIUS)
    np.testing.assert_array_equal(values, original)
