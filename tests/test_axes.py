"""Tests for the axis vocabulary and the centered-angular boundary convention.

``fftshift``/``ifftshift`` must appear nowhere in ``src/`` outside
``pypft.axes`` -- every other module reorders its angular axis through
``_center_angular``/``_uncenter_angular`` instead. The last test here is
that lint-as-test.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from pypft.axes import (
    DEFAULT_BATCH_AXIS,
    POLAR_SAMPLE_NDIM,
    PolarAxis,
    _center_angular,
    _uncenter_angular,
    _value_is_polar_sample_or_batch,
)

_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "pypft"
_FFTSHIFT_PATTERN = re.compile(r"\bi?fftshift\b")


def test_polar_axis_has_exactly_the_two_sample_axes():
    """``PolarAxis`` names the radial and angular axes only -- no batch member."""
    assert [axis.name for axis in PolarAxis] == ["RADIAL", "ANGULAR"]


def test_axis_values_are_numpy_axis_indices():
    """``PolarAxis`` members are literally the ``numpy`` axis they name."""
    assert (PolarAxis.RADIAL, PolarAxis.ANGULAR) == (0, 1)
    assert isinstance(PolarAxis.RADIAL, int)


def test_polar_sample_ndim_counts_the_sample_axes():
    """A single polar sample has one axis per ``PolarAxis`` member."""
    assert POLAR_SAMPLE_NDIM == len(PolarAxis) == 2


def test_default_batch_axis_is_the_axis_after_the_sample_axes():
    """On a batch, the only defaulted axis resolves to index ``POLAR_SAMPLE_NDIM``."""
    assert DEFAULT_BATCH_AXIS == -1
    n_dims = POLAR_SAMPLE_NDIM + 1
    assert (n_dims + DEFAULT_BATCH_AXIS) % n_dims == POLAR_SAMPLE_NDIM


@pytest.mark.parametrize(argnames="shape", argvalues=[(4, 5), (4, 5, 3)])
def test_a_single_sample_or_a_batch_is_accepted(shape: tuple[int, ...]) -> None:
    """Both a ``(radial, angular)`` sample and a batch of them pass."""
    _value_is_polar_sample_or_batch(value=np.zeros(shape=shape))


@pytest.mark.parametrize(argnames="shape", argvalues=[(4,), (4, 5, 3, 2)])
def test_any_other_rank_raises_naming_sample_and_batch(shape: tuple[int, ...]) -> None:
    """Every other rank raises, naming both accepted layouts."""
    match = r"a single sample \(radial, angular\) or a batch \(radial, angular, batch\)"
    with pytest.raises(ValueError, match=match):
        _value_is_polar_sample_or_batch(value=np.zeros(shape=shape))


def test_center_and_uncenter_angular_are_exact_inverses():
    """``_uncenter_angular`` undoes ``_center_angular`` bit-for-bit."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((5, 7, 3))

    centered = _center_angular(values=values, axis=PolarAxis.ANGULAR)
    restored = _uncenter_angular(values=centered, axis=PolarAxis.ANGULAR)

    assert np.array_equal(a1=restored, a2=values)


def test_center_angular_moves_index_zero_to_the_middle():
    """Index 0 (angle/harmonic 0 in natural order) lands at ``size // 2``."""
    size = 8
    natural = np.arange(size).reshape(1, size)  # a (radial, angular) row

    centered = _center_angular(values=natural, axis=PolarAxis.ANGULAR)

    assert centered[0, size // 2] == 0


@pytest.mark.parametrize(
    "path", sorted(_SRC_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(_SRC_DIR))
)
def test_fftshift_appears_only_in_the_axes_module(path):
    """No module but ``axes.py`` calls ``fftshift``/``ifftshift`` directly."""
    if path.name == "axes.py":
        pytest.skip("axes.py is the one place allowed to call fftshift/ifftshift")
    text = path.read_text(encoding="utf-8")
    assert not _FFTSHIFT_PATTERN.search(text), (
        f"{path} names fftshift/ifftshift directly; route through "
        "pypft.axes._center_angular/_uncenter_angular instead"
    )
