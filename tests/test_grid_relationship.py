"""Tests pinning how the transform's grid relates to a uniform polar grid.

``DESIGN_NOTES.md``, "Grid: the spatial row index is a spoke, and the transform
identifies it with a harmonic," states these facts and records the measurements they
come from; ``notebooks/02_sampling_grids.ipynb`` shows them. Every check runs on a
small grid so the suite stays fast, and every tolerance is the value measured on that
grid with a stated margin:

- Same spokes, different radii: one radial index sits at a different radius on
  every spoke, and the spacing near the outer edge settles to ``pi * R / j_{|p|N1}``.
- Spurious harmonics: on-grid samples of a circularly symmetric function have a
  non-zero harmonic ``+-1``, although the DHT of the exact harmonic-0 samples is
  accurate to rounding error.
- Per-spoke interpolation: interpolating uniform radii along each spoke onto
  ``grid.r`` gives the same ``forward_pft`` result as sampling on the grid.
- The average dB error hides the error: a case with a very low ``E_avg`` whose
  relative L2 error exceeds the size of the transform itself.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.special import jn_zeros

from pypft.dft import angular_dft
from pypft.dht import hankel_transform
from pypft.grid import PolarGrid
from pypft.transform import forward_pft

# ======================================================================================
# Shared grid, test functions and error measures
# ======================================================================================

#: A small grid on which every effect below is clearly measurable.
_GRID = PolarGrid(n_radial=64, n_angular=15, R=10.0)

#: The center of the off-center Gaussian.
_CENTER = (2.0, -1.0)

#: The radial index whose radius is compared across spokes.
_RADIAL_INDEX = 10


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


def _gaussian_transform(
    rho: np.ndarray, psi: np.ndarray, center: tuple[float, float]
) -> np.ndarray:
    """Evaluate the exact 2-D Fourier transform of ``_gaussian``.

    :param rho: The radial frequencies.
    :type rho: np.ndarray
    :param psi: The frequency angles, broadcastable against ``rho``.
    :type psi: np.ndarray
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: ``pi * exp(-rho**2 / 4) * exp(-i rho . center)`` at every point.
    :rtype: np.ndarray

    """
    x0, y0 = center
    phase = rho * (np.cos(psi) * x0 + np.sin(psi) * y0)
    return np.pi * np.exp(-(rho**2) / 4) * np.exp(-1j * phase)


def _on_grid_samples(grid: PolarGrid, center: tuple[float, float]) -> np.ndarray:
    """Sample ``_gaussian`` on ``grid``, in PyPFT's ``(radial, angular)`` layout.

    :param grid: The grid to sample on.
    :type grid: PolarGrid
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: An ``(n_radial, n_angular)`` complex array.
    :rtype: np.ndarray

    """
    return _gaussian(r=grid.r.T, theta=grid.theta[np.newaxis, :], center=center) + 0j


def _exact_transform(grid: PolarGrid, center: tuple[float, float]) -> np.ndarray:
    """Evaluate ``_gaussian_transform`` on ``grid``'s frequency points.

    :param grid: The grid to evaluate on.
    :type grid: PolarGrid
    :param center: The ``(x, y)`` center of the Gaussian.
    :type center: tuple[float, float]
    :returns: An ``(n_radial, n_angular)`` complex array.
    :rtype: np.ndarray

    """
    return _gaussian_transform(
        rho=grid.rho.T, psi=grid.psi[np.newaxis, :], center=center
    )


def _relative_l2(approximation: np.ndarray, exact: np.ndarray) -> float:
    """The relative L2 error ``||approximation - exact|| / ||exact||``.

    :param approximation: The computed values.
    :type approximation: np.ndarray
    :param exact: The exact values.
    :type exact: np.ndarray
    :returns: The relative L2 error.
    :rtype: float

    """
    return float(np.linalg.norm(x=approximation - exact) / np.linalg.norm(x=exact))


def _mean_db(approximation: np.ndarray, exact: np.ndarray) -> float:
    """The average dB error ``E_avg``, relative to the peak of ``approximation``.

    :param approximation: The computed values.
    :type approximation: np.ndarray
    :param exact: The exact values.
    :type exact: np.ndarray
    :returns: The mean of ``20 log10(|approximation - exact| / max|approximation|)``.
    :rtype: float

    """
    error = np.abs(approximation - exact) / np.max(a=np.abs(approximation))
    return float(np.mean(a=20 * np.log10(error)))


# ======================================================================================
# Same spokes, different radii
# ======================================================================================

#: Measured spread of ``grid.r[:, _RADIAL_INDEX]`` on ``_GRID``: 1.66 to 2.06. The
#: test only requires the spokes to differ by more than this fraction of the
#: smallest radius, a quarter of the measured 24%.
_MIN_RADIUS_SPREAD = 0.06

#: Measured largest relative difference between the outermost spacing and
#: ``pi * R / j_{|p|N1}`` on ``_GRID``: 5.6e-4, allowed here with a margin of ~3.5x.
_OUTER_SPACING_RTOL = 2e-3


def test_one_radial_index_sits_at_a_different_radius_on_each_spoke() -> None:
    """A row of constant radial index is not a ring."""
    radii = _GRID.r[:, _RADIAL_INDEX]
    spread = (radii.max() - radii.min()) / radii.min()
    assert spread > _MIN_RADIUS_SPREAD
    # Only spokes p and -p share their radii; every other pair differs.
    orders = np.abs(_GRID.harmonics)
    for i in range(_GRID.n_angular):
        for j in range(_GRID.n_angular):
            if orders[i] != orders[j]:
                assert radii[i] != radii[j]


def test_outer_spacing_settles_to_pi_r_over_the_last_zero() -> None:
    """Near ``R`` the spacing along spoke ``p`` approaches ``pi * R / j_{|p|N1}``."""
    outer_spacing = _GRID.r[:, -1] - _GRID.r[:, -2]
    limits = np.array(
        object=[
            np.pi * _GRID.R / jn_zeros(n=int(abs(p)), nt=_GRID.n_radial + 1)[-1]
            for p in _GRID.harmonics
        ]
    )
    np.testing.assert_allclose(
        actual=outer_spacing, desired=limits, rtol=_OUTER_SPACING_RTOL
    )


# ======================================================================================
# Spurious harmonics of a circularly symmetric function
# ======================================================================================

#: Measured ``|f_{+-1}|`` of the centered Gaussian's on-grid samples on ``_GRID``:
#: 0.077 (harmonic 0 peaks at 0.880 instead of 1). A circularly symmetric function
#: has no harmonic 1 at all, so anything above this threshold is the grid's doing.
_MIN_SPURIOUS_HARMONIC = 0.05

#: Measured error of the DHT of the exact harmonic-0 samples on ``_GRID``: 2.8e-16
#: relative to the peak ``pi``.
_EXACT_HARMONIC_DHT_ATOL = 1e-13


def test_on_grid_samples_of_a_circular_function_have_spurious_harmonics() -> None:
    """The angular DFT of on-grid samples of ``exp(-r**2)`` has a harmonic ``+-1``."""
    samples = _on_grid_samples(grid=_GRID, center=(0.0, 0.0))
    # Divided by n_angular, each entry is the amplitude of that harmonic.
    coefficients = angular_dft(x=samples, axis=1) / _GRID.n_angular
    amplitude = np.max(a=np.abs(coefficients), axis=0)
    zero = _GRID.n_angular // 2
    assert amplitude[zero + 1] > _MIN_SPURIOUS_HARMONIC
    assert amplitude[zero - 1] > _MIN_SPURIOUS_HARMONIC


def test_dht_of_the_exact_harmonic_zero_samples_is_accurate() -> None:
    """The DHT is not the source of the error: harmonic 0 alone transforms exactly."""
    zero = _GRID.n_angular // 2
    radii, frequencies = _GRID.r[zero], _GRID.rho[zero]
    transform = 2 * np.pi * hankel_transform(f=np.exp(-(radii**2)), n=0, R=_GRID.R)
    exact = np.pi * np.exp(-(frequencies**2) / 4)
    np.testing.assert_allclose(
        actual=transform / np.pi,
        desired=exact / np.pi,
        rtol=0.0,
        atol=_EXACT_HARMONIC_DHT_ATOL,
    )


# ======================================================================================
# Interpolating uniform radii along each spoke
# ======================================================================================

#: Measured relative L2 difference, on ``_GRID``, between ``forward_pft`` of the
#: spline-interpolated samples and ``forward_pft`` of the on-grid samples: 4.2e-6
#: off-center (1.1e-5 centered). Allowed here with a margin of ~10x over the larger.
_INTERPOLATION_RTOL = 1e-4


@pytest.mark.parametrize(argnames="center", argvalues=[(0.0, 0.0), _CENTER])
def test_per_spoke_interpolation_matches_on_grid_sampling(
    center: tuple[float, float],
) -> None:
    """Uniform radii, interpolated along each spoke onto ``grid.r``, act as on-grid."""
    # Uniform radii from 0 to R, as a uniform polar image would hold them.
    knots = np.linspace(start=0.0, stop=_GRID.R, num=_GRID.n_radial + 1)
    interpolated = np.stack(
        arrays=[
            CubicSpline(x=knots, y=_gaussian(r=knots, theta=angle, center=center))(
                _GRID.r[row]
            )
            for row, angle in enumerate(_GRID.theta)
        ],
        axis=1,
    )
    from_uniform = forward_pft(f=interpolated + 0j, grid=_GRID)
    on_grid = forward_pft(f=_on_grid_samples(grid=_GRID, center=center), grid=_GRID)
    assert _relative_l2(approximation=from_uniform, exact=on_grid) < (
        _INTERPOLATION_RTOL
    )


# ======================================================================================
# The average dB error hides the error
# ======================================================================================

#: An off-center Gaussian on a grid with many spokes and a small ``R``. Measured:
#: ``E_avg`` -94.5 dB, relative L2 error 2.14.
_HIDDEN_ERROR_GRID = PolarGrid(n_radial=64, n_angular=31, R=5.0)

#: The ``E_avg`` the case must stay below: far better than the -60 dB accuracy
#: target ``pypft.grid.check_adequacy`` uses, with a 14.5 dB margin to the
#: measured value.
_HIDDEN_ERROR_MAX_DB = -80.0

#: The relative L2 error the case must exceed: half the measured 2.14, and ten times
#: the 0.1 that already rules out an approximation.
_HIDDEN_ERROR_MIN_RELATIVE_L2 = 1.0


def test_average_db_error_cannot_certify_accuracy() -> None:
    """``E_avg`` far below -60 dB coexists with a relative L2 error above 100%."""
    computed = forward_pft(
        f=_on_grid_samples(grid=_HIDDEN_ERROR_GRID, center=_CENTER),
        grid=_HIDDEN_ERROR_GRID,
    )
    exact = _exact_transform(grid=_HIDDEN_ERROR_GRID, center=_CENTER)
    assert _mean_db(approximation=computed, exact=exact) < _HIDDEN_ERROR_MAX_DB
    assert _relative_l2(approximation=computed, exact=exact) > (
        _HIDDEN_ERROR_MIN_RELATIVE_L2
    )
