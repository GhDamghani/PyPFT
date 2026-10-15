"""Tests for ``scripts/make_md_cnn_excerpt.py``, the cutter of the MD-CNN fixture.

The full dataset file is not part of the repository, so the committed fixture is
reproduced only where it is present (``.local_files/``); everything else runs on a
small synthetic array in the dataset's layout.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.io import loadmat

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "make_md_cnn_excerpt.py"


def _load_script() -> ModuleType:
    """Import ``scripts/make_md_cnn_excerpt.py`` (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name="make_md_cnn_excerpt", location=_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module)
    return module


make_md_cnn_excerpt = _load_script()


def _data() -> np.ndarray:
    """A ``(2, 3, 12, 4, 5, 2)`` array in the dataset's layout, coil 3 the strongest.

    :returns: The array.
    :rtype: np.ndarray

    """
    data = np.random.default_rng(seed=0).standard_normal(size=(2, 3, 12, 4, 5, 2))
    data[..., 3, :] *= 10.0
    data[..., 1, :] *= 5.0
    return data


def test_the_excerpt_keeps_the_layout_and_the_center() -> None:
    data = _data()
    excerpt = make_md_cnn_excerpt.make_excerpt(data=data, readout=8, coils=2)
    assert excerpt.shape == (1, 1, 8, 4, 2, 2)
    assert excerpt.dtype == np.float32
    # The full readout's center, index 6, is the excerpt's center, index 4; the two
    # strongest coils, 1 and 3, in coil order.
    np.testing.assert_array_equal(
        excerpt[0, 0, 4], data[0, 0, 6][:, [1, 3]].astype(np.float32)
    )


@pytest.mark.parametrize(
    ("readout", "coils"), [(7, 2), (14, 2), (8, 6), (0, 2), (8, 0)]
)
def test_invalid_sizes_are_rejected(readout: int, coils: int) -> None:
    with pytest.raises(ValueError):
        make_md_cnn_excerpt.make_excerpt(data=_data(), readout=readout, coils=coils)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"data": [[1.0]]}, TypeError),
        ({"data": np.ones((2, 3, 12, 4, 5))}, ValueError),
        ({"readout": 8.0}, TypeError),
        ({"coils": "2"}, TypeError),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, error: type) -> None:
    arguments = {"data": _data(), "readout": 8, "coils": 2, **kwargs}
    with pytest.raises(error):
        make_md_cnn_excerpt.make_excerpt(**arguments)


@pytest.mark.skipif(
    not make_md_cnn_excerpt.DEFAULT_SOURCE.is_file(),
    reason="the full dataset file is not in .local_files/",
)
def test_the_defaults_reproduce_the_committed_fixture(tmp_path: Path) -> None:
    output = tmp_path / "excerpt.mat"
    make_md_cnn_excerpt.main(argv=["--output", str(output)])
    variable = make_md_cnn_excerpt.VARIABLE
    np.testing.assert_array_equal(
        loadmat(file_name=output)[variable],
        loadmat(file_name=make_md_cnn_excerpt.DEFAULT_OUTPUT)[variable],
    )
