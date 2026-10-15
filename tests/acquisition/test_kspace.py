"""Tests for ``pypft.acquisition``'s layout-independent layer.

``radial_lines_to_polar`` splits k-space lines through the center into PyPFT's uniform
polar spokes, and ``RadialKSpace`` carries the result. These tests build lines from an
analytic 2-D Fourier transform, so every converted sample has a known value, and pin
the layout, every accepted line ordering, the rejected trajectories, the radius, the
file round trip, and the result's fitness for both paths (``inverse_pft_ring`` with
``radius`` and ``resample_uniform_polar_frequency``). Every tolerance is a measured
value with a stated margin.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypft.acquisition import (
    NonUniformAnglesError,
    RadialKSpace,
    radial_lines_to_polar,
)
from pypft.dft import harmonics
from pypft.grid import PolarGrid, resample_uniform_polar_frequency
from pypft.ring import inverse_pft_ring

# ======================================================================================
# An analytic acquisition
# ======================================================================================

#: Readout samples per line, and the center's index among them.
_N_READOUT, _CENTER = 128, 64

#: Lines of the default acquisition, at uniform angles over half a turn.
_N_LINES = 64

#: The off-center Gaussian ``exp(-|x - center|**2 / width**2)`` the k-space comes from.
_GAUSSIAN_CENTER, _GAUSSIAN_WIDTH = (6.0, -3.0), 4.0


def _gaussian_transform(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """The 2-D Fourier transform of the off-center Gaussian, PyPFT's convention.

    :param kx: The ``x`` frequencies, in radians per pixel.
    :type kx: np.ndarray
    :param ky: The ``y`` frequencies, in radians per pixel.
    :type ky: np.ndarray
    :returns: ``pi w**2 exp(-w**2 |k|**2 / 4) exp(-i k . center)``.
    :rtype: np.ndarray

    """
    x0, y0 = _GAUSSIAN_CENTER
    width = _GAUSSIAN_WIDTH
    envelope = np.pi * width**2 * np.exp(-(width**2) * (kx**2 + ky**2) / 4)
    return envelope * np.exp(-1j * (kx * x0 + ky * y0))


def _gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The off-center Gaussian itself.

    :param x: The ``x`` coordinates, in pixels.
    :type x: np.ndarray
    :param y: The ``y`` coordinates, in pixels.
    :type y: np.ndarray
    :returns: The Gaussian's values.
    :rtype: np.ndarray

    """
    x0, y0 = _GAUSSIAN_CENTER
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / _GAUSSIAN_WIDTH**2)


def _lines(
    angles: np.ndarray, n_readout: int = _N_READOUT, center: int = _CENTER
) -> np.ndarray:
    """Acquire the Gaussian's k-space along lines at ``angles``.

    :param angles: Each line's angle, in radians.
    :type angles: np.ndarray
    :param n_readout: Readout samples per line.
    :type n_readout: int
    :param center: The center's readout index.
    :type center: int
    :returns: The lines, ``(readout, line)``.
    :rtype: np.ndarray

    """
    # Readout sample s sits at (s - center) * 2 pi / n_readout radians per pixel.
    frequency = (np.arange(n_readout) - center)[:, np.newaxis] * (2 * np.pi / n_readout)
    return _gaussian_transform(
        kx=frequency * np.cos(angles), ky=frequency * np.sin(angles)
    )


def _uniform_angles(n_lines: int = _N_LINES) -> np.ndarray:
    """Sequential uniform line angles over half a turn.

    :param n_lines: The number of lines.
    :type n_lines: int
    :returns: ``l * pi / n_lines``.
    :rtype: np.ndarray

    """
    return np.arange(n_lines) * (np.pi / n_lines)


# ======================================================================================
# Layout
# ======================================================================================


def test_every_sample_lands_at_its_own_frequency_and_angle() -> None:
    kspace = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    n_radial, n_angular = kspace.values.shape
    assert (n_radial, n_angular) == (_CENTER, 2 * _N_LINES)
    # The centered spokes of a uniform polar array, and its uniform radii.
    spoke_angles = harmonics(n_angular) * (2 * np.pi / n_angular)
    radii = np.arange(n_radial) * (kspace.radius / n_radial)
    expected = _gaussian_transform(
        kx=radii[:, np.newaxis] * np.cos(spoke_angles),
        ky=radii[:, np.newaxis] * np.sin(spoke_angles),
    )
    # Measured 1.7e-14 relative: the same values, computed through other products.
    np.testing.assert_allclose(actual=kspace.values, desired=expected, rtol=1e-12)


def test_radius_is_the_readout_frequency_of_the_kept_samples() -> None:
    kspace = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    assert kspace.radius == pytest.approx(_CENTER * 2 * np.pi / _N_READOUT)


@pytest.mark.parametrize(
    "order",
    [
        np.arange(_N_LINES)[::-1],
        # Two interleaved halves, as a segmented acquisition orders them.
        np.concatenate((np.arange(0, _N_LINES, 2), np.arange(1, _N_LINES, 2))),
        np.random.default_rng(seed=0).permutation(_N_LINES),
    ],
)
def test_any_line_order_gives_the_same_spokes(order: np.ndarray) -> None:
    sequential = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    angles = _uniform_angles()[order]
    reordered = radial_lines_to_polar(
        lines=_lines(angles=angles), angles=angles, center_index=_CENTER
    )
    np.testing.assert_array_equal(reordered.values, sequential.values)


def test_a_line_read_the_other_way_gives_the_same_spokes() -> None:
    sequential = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    # Every odd line acquired in the opposite direction, half a turn round, and
    # every angle shifted by whole turns: still the same set of lines.
    angles = _uniform_angles() + np.where(np.arange(_N_LINES) % 2, np.pi, -2 * np.pi)
    flipped = radial_lines_to_polar(
        lines=_lines(angles=angles), angles=angles, center_index=_CENTER
    )
    # Measured 2.2e-14 relative: each opposite-direction sample is the same
    # frequency computed through another angle's cosine and sine.
    np.testing.assert_allclose(
        actual=flipped.values, desired=sequential.values, rtol=1e-12, atol=1e-12
    )


def test_an_off_center_readout_keeps_the_shorter_side() -> None:
    # An asymmetric echo: 40 samples before the center, 87 after it.
    kspace = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles(), center=40),
        angles=_uniform_angles(),
        center_index=40,
    )
    assert kspace.values.shape == (41, 2 * _N_LINES)
    assert kspace.radius == pytest.approx(41 * 2 * np.pi / _N_READOUT)


def test_batches_and_real_values_keep_their_layout_and_type() -> None:
    lines = np.random.default_rng(seed=1).standard_normal(size=(16, 4, 3))
    kspace = radial_lines_to_polar(
        lines=lines, angles=_uniform_angles(n_lines=4), center_index=8
    )
    assert kspace.values.shape == (8, 8, 3)
    assert kspace.values.dtype == lines.dtype
    single = radial_lines_to_polar(
        lines=lines[..., 1], angles=_uniform_angles(n_lines=4), center_index=8
    )
    np.testing.assert_array_equal(kspace.values[..., 1], single.values)


# ======================================================================================
# Rejected trajectories and input
# ======================================================================================


def test_golden_angle_lines_are_rejected() -> None:
    golden = np.arange(_N_LINES) * np.deg2rad(111.246117975)
    with pytest.raises(NonUniformAnglesError, match="multiple"):
        radial_lines_to_polar(
            lines=_lines(angles=golden), angles=golden, center_index=_CENTER
        )
    assert issubclass(NonUniformAnglesError, ValueError)


def test_lines_repeating_an_angle_modulo_half_a_turn_are_rejected() -> None:
    angles = _uniform_angles()
    angles[1] = angles[0] + np.pi
    with pytest.raises(NonUniformAnglesError, match="distinct"):
        radial_lines_to_polar(
            lines=_lines(angles=angles), angles=angles, center_index=_CENTER
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"lines": [[1.0]]}, TypeError),
        ({"lines": np.ones(16)}, ValueError),
        ({"lines": np.ones((16, 4, 2, 2))}, ValueError),
        ({"lines": np.full((16, 4), np.nan)}, ValueError),
        ({"angles": np.zeros(3)}, ValueError),
        ({"angles": np.zeros((4, 1))}, ValueError),
        ({"angles": [0.0, 1.0, 2.0, 3.0]}, TypeError),
        ({"center_index": 16}, ValueError),
        ({"center_index": -1}, ValueError),
        ({"center_index": 8.0}, TypeError),
    ],
)
def test_invalid_input_is_rejected(kwargs: dict, error: type) -> None:
    arguments = {
        "lines": np.ones((16, 4)),
        "angles": _uniform_angles(n_lines=4),
        "center_index": 8,
        **kwargs,
    }
    with pytest.raises(error):
        radial_lines_to_polar(**arguments)


# ======================================================================================
# RadialKSpace
# ======================================================================================


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"values": [[1.0]]}, TypeError),
        ({"values": np.ones(4)}, ValueError),
        ({"values": np.full((4, 4), np.inf)}, ValueError),
        ({"radius": 1}, TypeError),
        ({"radius": 0.0}, ValueError),
    ],
)
def test_radial_kspace_validates_its_fields(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        RadialKSpace(**{"values": np.ones((4, 4)), "radius": 1.0, **kwargs})


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    values = np.random.default_rng(seed=2).standard_normal(size=(8, 6, 2)) + 1j
    path = tmp_path / "kspace.npz"
    RadialKSpace(values=values, radius=2.5).save(path=path)
    loaded = RadialKSpace.load(path=path)
    np.testing.assert_array_equal(loaded.values, values)
    assert loaded.radius == 2.5


def test_save_and_load_reject_invalid_paths(tmp_path: Path) -> None:
    kspace = RadialKSpace(values=np.ones((4, 4)), radius=1.0)
    with pytest.raises(ValueError, match="suffix"):
        kspace.save(path=tmp_path / "kspace.npy")
    with pytest.raises(TypeError):
        kspace.save(path=str(tmp_path / "kspace.npz"))  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        RadialKSpace.load(path=tmp_path / "missing.npz")
    np.savez(tmp_path / "other.npz", values=np.ones((4, 4)))
    with pytest.raises(ValueError, match="radius"):
        RadialKSpace.load(path=tmp_path / "other.npz")


# ======================================================================================
# Into the transform
# ======================================================================================

#: A grid whose frequency samples all lie inside the acquisition's radius ``pi``.
_GRID = PolarGrid(n_radial=40, n_angular=31, R=48.0)


def test_converted_kspace_feeds_the_ring_route() -> None:
    kspace = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    f = inverse_pft_ring(F=kspace.values, grid=_GRID, radius=kspace.radius)
    r, theta = _GRID.r.T, _GRID.theta[np.newaxis, :]
    expected = _gaussian(x=r * np.cos(theta), y=r * np.sin(theta))
    error = np.linalg.norm(f - expected) / np.linalg.norm(expected)
    # Measured 2.4e-5 relative L2.
    assert error < 1e-4


def test_converted_kspace_feeds_the_exact_path_resampler() -> None:
    kspace = radial_lines_to_polar(
        lines=_lines(angles=_uniform_angles()),
        angles=_uniform_angles(),
        center_index=_CENTER,
    )
    # The grid's 31 spokes are fewer than the data's 128: no upsampling warning.
    with warnings.catch_warnings():
        warnings.simplefilter(action="error")
        F = resample_uniform_polar_frequency(
            values=kspace.values, grid=_GRID, radius=kspace.radius
        )
    rho, psi = _GRID.rho.T, _GRID.psi[np.newaxis, :]
    expected = _gaussian_transform(kx=rho * np.cos(psi), ky=rho * np.sin(psi))
    error = np.linalg.norm(F - expected) / np.linalg.norm(expected)
    # Measured 4.2e-5 relative L2 (radial and periodic angular splines).
    assert error < 2e-4
