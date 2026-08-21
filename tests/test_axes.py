"""Tests for the axis vocabulary and the centered-angular boundary convention.

develop_plan.md §3.9 requires ``fftshift``/``ifftshift`` to appear nowhere in
``src/`` outside ``pypft.axes`` -- every other module reorders its angular
axis through ``_center_angular``/``_uncenter_angular`` instead. The last test
here is that lint-as-test.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from pypft.axes import DEFAULT_BATCH_AXIS, Axis, _center_angular, _uncenter_angular

_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "pypft"
_FFTSHIFT_PATTERN = re.compile(r"\bi?fftshift\b")


def test_axis_values_are_numpy_axis_indices():
    """``Axis`` members are literally the ``numpy`` axis they name."""
    assert (Axis.RADIAL, Axis.ANGULAR, Axis.BATCH) == (0, 1, 2)
    assert isinstance(Axis.RADIAL, int)


def test_default_batch_axis_matches_the_batch_axis_enum_member():
    """The only defaulted axis resolves to ``Axis.BATCH`` on a 3-D array."""
    assert DEFAULT_BATCH_AXIS == -1
    n_dims = 3
    assert (n_dims + DEFAULT_BATCH_AXIS) % n_dims == Axis.BATCH


def test_center_and_uncenter_angular_are_exact_inverses():
    """``_uncenter_angular`` undoes ``_center_angular`` bit-for-bit."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal((5, 7, 3))

    centered = _center_angular(values, axis=Axis.ANGULAR)
    restored = _uncenter_angular(centered, axis=Axis.ANGULAR)

    assert np.array_equal(restored, values)


def test_center_angular_moves_index_zero_to_the_middle():
    """Index 0 (angle/harmonic 0 in natural order) lands at ``size // 2``."""
    size = 8
    natural = np.arange(size).reshape(1, size)  # a (radial, angular) row

    centered = _center_angular(natural, axis=Axis.ANGULAR)

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
