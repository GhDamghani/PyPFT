"""Tests for the transform's own sampling grid (``pypft.grid``).

Three concerns are checked: that ``PolarGrid.r``/``.rho`` reproduce the MATLAB
appendix's own space-limited formulas (Eqs. A-2/A-4) independently of
``pypft.dht``'s internals, that the frozen/hashable value-object contract holds, and
that ``check_adequacy``/``check_nyquist_adequacy`` warn exactly when the underlying
measured/analytical relationships say they should -- both are load-bearing because
``pytest.ini``'s ``filterwarnings = ["error"]`` turns any unexpected ``warnings.warn``
into a test failure on its own.
"""

import dataclasses

import numpy as np
import pytest
from scipy.special import jn_zeros

from pypft.dft import harmonics
from pypft.dht import sample_points
from pypft.grid import (
    AdequacyWarning,
    LimitKind,
    NyquistWarning,
    PolarGrid,
    check_adequacy,
    check_nyquist_adequacy,
    sample_cartesian,
)

_R = 40.0
_N_RADIAL = 383


def _manual_space_limited_r_rho(
    n_angular: int, n_radial: int, R: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the MATLAB appendix's A-2/A-4 formulas directly from Bessel zeros.

    Independent of ``pypft.dht``/``pypft.grid`` -- built straight from
    ``scipy.special.jn_zeros`` -- so this is an external oracle, not a
    reimplementation of the code under test.
    """
    harms = harmonics(n_angular=n_angular)
    r = np.empty((n_angular, n_radial))
    rho = np.empty((n_angular, n_radial))
    for row, p in enumerate(harms):
        zeros = jn_zeros(n=int(abs(p)), nt=n_radial + 1)
        j_pn1 = zeros[-1]
        j_pk = zeros[:-1]
        r[row, :] = j_pk / j_pn1 * R
        rho[row, :] = j_pk / R
    return r, rho


@pytest.mark.parametrize("n_angular", [15, 16])
def test_r_and_rho_match_the_appendix_formulas(n_angular):
    """``PolarGrid.r``/``.rho`` match the appendix's A-2/A-4 formulas."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=n_angular, R=_R)
    expected_r, expected_rho = _manual_space_limited_r_rho(
        n_angular=n_angular, n_radial=_N_RADIAL, R=_R
    )
    # A tight allclose, not exact equality: the appendix's own r = j/jN*R and this
    # test's r = j/jN*R associate the multiply/divide in the same order but through a
    # different call path, which floating-point arithmetic is not guaranteed to
    # reproduce bit-for-bit.
    np.testing.assert_allclose(grid.r, expected_r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(grid.rho, expected_rho, rtol=1e-13, atol=1e-13)


def test_r_at_harmonic_zero_matches_dht_sample_points_exactly():
    """The key cross-check: harmonic 0's row is ``pypft.dht.sample_points``'s own r."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=15, R=_R)
    zero_row = int(np.where(grid.harmonics == 0)[0][0])
    expected_r, expected_rho = sample_points(n=0, size=_N_RADIAL, R=_R)
    np.testing.assert_array_equal(grid.r[zero_row, :], expected_r)
    np.testing.assert_array_equal(grid.rho[zero_row, :], expected_rho)


def test_band_limited_swaps_r_and_rho_relative_to_space_limited():
    """A band-limited grid's ``r``/``rho`` are the space-limited grid's, swapped."""
    space = PolarGrid(
        n_radial=_N_RADIAL, n_angular=15, R=_R, limit_kind=LimitKind.SPACE_LIMITED
    )
    band = PolarGrid(
        n_radial=_N_RADIAL, n_angular=15, R=_R, limit_kind=LimitKind.BAND_LIMITED
    )
    np.testing.assert_array_equal(band.r, space.rho)
    np.testing.assert_array_equal(band.rho, space.r)


@pytest.mark.parametrize("n_angular", [15, 16])
def test_theta_and_psi_are_identical_and_centered(n_angular):
    """``theta``/``psi`` agree and place harmonic 0 at angle 0."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=n_angular, R=_R)
    np.testing.assert_array_equal(grid.theta, grid.psi)
    zero_row = int(np.where(grid.harmonics == 0)[0][0])
    assert grid.theta[zero_row] == pytest.approx(0.0)
    np.testing.assert_allclose(grid.theta, grid.harmonics * (2 * np.pi / n_angular))


def test_polar_grid_is_frozen():
    """Fields cannot be reassigned after construction."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=15, R=_R)
    with pytest.raises(dataclasses.FrozenInstanceError):
        grid.R = 1.0  # type: ignore[misc]


def test_polar_grid_is_hashable_and_usable_as_a_dict_key():
    """Equal grids hash equal, so a grid can key a cache the way ``(n, size)`` does."""
    grid_a = PolarGrid(n_radial=_N_RADIAL, n_angular=15, R=_R)
    grid_b = PolarGrid(n_radial=_N_RADIAL, n_angular=15, R=_R)
    grid_c = PolarGrid(n_radial=_N_RADIAL, n_angular=16, R=_R)
    assert grid_a == grid_b
    assert hash(grid_a) == hash(grid_b)
    cache = {grid_a: "cached"}
    assert cache[grid_b] == "cached"
    assert grid_c not in cache


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_radial", 1.0),
        ("n_angular", 1.0),
        ("R", 1),
        ("limit_kind", "space"),
    ],
)
def test_polar_grid_rejects_wrong_field_types(field, value):
    """Every field is type-validated in ``__post_init__``."""
    kwargs = {"n_radial": _N_RADIAL, "n_angular": 15, "R": _R}
    kwargs[field] = value
    with pytest.raises(TypeError):
        PolarGrid(**kwargs)


def test_check_adequacy_is_silent_for_a_grid_matching_the_pft_forward_gate():
    """No warning for ``N2=15, N1=383`` -- the forward gate's own parameters."""
    check_adequacy(
        grid=PolarGrid(n_radial=383, n_angular=15, R=_R)
    )  # filterwarnings=error catches any warning


def test_check_adequacy_warns_for_a_grid_with_positive_measured_e_max():
    """Warns for ``N2=64, N1=383``, the combination measured to give positive E_max."""
    with pytest.warns(AdequacyWarning, match="n_radial"):
        check_adequacy(grid=PolarGrid(n_radial=383, n_angular=64, R=_R))


def test_check_nyquist_adequacy_is_silent_within_the_oracle_band_limit():
    """No warning at the paper's own ``Wr=30`` band limit for ``N1=383, R=40``."""
    check_nyquist_adequacy(
        grid=PolarGrid(n_radial=383, n_angular=15, R=_R), band_limit=30.0
    )


def test_check_nyquist_adequacy_warns_when_n_radial_is_too_small():
    """Warns when ``band_limit`` is far larger than ``N1`` can support."""
    with pytest.warns(NyquistWarning, match="Nyquist"):
        check_nyquist_adequacy(
            grid=PolarGrid(n_radial=383, n_angular=15, R=_R), band_limit=1000.0
        )


def test_check_nyquist_adequacy_rejects_band_limited_grids():
    """The Nyquist helper only handles the space-limited case."""
    grid = PolarGrid(
        n_radial=383, n_angular=15, R=_R, limit_kind=LimitKind.BAND_LIMITED
    )
    with pytest.raises(NotImplementedError):
        check_nyquist_adequacy(grid=grid, band_limit=30.0)


def _pixel_offsets(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered column/row pixel offsets for a square image, as ``float64``."""
    center = size // 2
    rows, cols = np.mgrid[0:size, 0:size]
    return (cols - center).astype(np.float64), (rows - center).astype(np.float64)


def test_sample_cartesian_reproduces_a_plane_function_at_the_grid_points():
    """``sample_cartesian`` samples close to where ``grid.r``/``grid.theta`` say.

    A plane ``dx + 2*dy`` is (in exact arithmetic) recovered by bilinear
    interpolation at any real point, so comparing against the analytically expected
    value at each grid point catches both a wrong radius and a wrong (or
    swapped/flipped) angle -- the coefficients ``1``/``2`` differ specifically so an
    ``x``/``y`` mix-up would not cancel out. The tolerance is loose (matching
    ``tests/test_geometry.py``'s own round-trip check) because ``cv2.remap``'s
    ``INTER_LINEAR`` interpolates on a fixed-point, 1/32-pixel sub-pixel grid rather
    than at full ``float64`` precision.
    """
    image_size = 256
    dx, dy = _pixel_offsets(image_size)
    image = dx + 2.0 * dy

    grid = PolarGrid(
        n_radial=50, n_angular=15, R=40.0
    )  # R << image_size / 2: stays inside the image
    sampled = sample_cartesian(image=image, grid=grid)

    expected = grid.r * np.cos(grid.theta)[:, np.newaxis] + 2.0 * (
        grid.r * np.sin(grid.theta)[:, np.newaxis]
    )
    error = np.abs(sampled - expected)
    assert error.mean() < 0.02
    assert error.max() < 0.1
