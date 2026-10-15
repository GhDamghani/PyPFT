"""Tests for ``pypft.acquisition``'s file layer: arrays read whatever their layout.

A ``.npy`` file's single array is read as it is; a ``.npz`` or ``.mat`` file's arrays
are listed from their headers and one is read by name. The same array read from any of
the three formats converts to the same k-space, which is what makes a
``RadialLineLayout`` independent of the file format.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pypft.acquisition import (
    ARRAY_FILE_SUFFIXES,
    RadialLineLayout,
    holds_several_arrays,
    load_stored_array,
    select_radial_kspace,
    stored_arrays,
)

_LAYOUT = RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX

#: A small array in ``_LAYOUT``: ``(slices, phases, readout, lines, coils, 2)``.
_SHAPE = (1, 2, 8, 3, 2, 2)

#: The name the multi-array files store the lines under, and of a second array.
_LINES_NAME, _OTHER_NAME = "kspace", "noise"


@pytest.fixture(name="lines")
def _lines() -> np.ndarray:
    """Build a small array in ``_LAYOUT``, every sample distinct."""
    return np.random.default_rng(seed=1).standard_normal(size=_SHAPE)


def _save(path: Path, lines: np.ndarray) -> Path:
    """Store ``lines`` in ``path``'s format: alone, or beside a second array."""
    other = np.zeros(shape=(4,))
    if path.suffix == ".npy":
        np.save(file=path, arr=lines)
    elif path.suffix == ".npz":
        np.savez(path, **{_LINES_NAME: lines, _OTHER_NAME: other})
    else:
        savemat(file_name=path, mdict={_LINES_NAME: lines, _OTHER_NAME: other})
    return path


# ======================================================================================
# Formats
# ======================================================================================


def test_the_formats_are_single_or_several_arrays(tmp_path: Path) -> None:
    assert ARRAY_FILE_SUFFIXES == (".npy", ".npz", ".mat")
    assert not holds_several_arrays(path=tmp_path / "a.npy")
    assert holds_several_arrays(path=tmp_path / "a.NPZ")
    assert holds_several_arrays(path=tmp_path / "a.mat")
    with pytest.raises(ValueError, match="suffix"):
        holds_several_arrays(path=tmp_path / "a.png")
    with pytest.raises(TypeError):
        holds_several_arrays(path="a.npy")  # type: ignore[arg-type]


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_stored_arrays_lists_names_and_shapes(
    tmp_path: Path, lines: np.ndarray, suffix: str
) -> None:
    path = _save(path=tmp_path / f"lines{suffix}", lines=lines)
    arrays = stored_arrays(path=path)
    # A MATLAB file stores a 1-D array as a row.
    other_shape = (4,) if suffix == ".npz" else (1, 4)
    assert arrays == {_LINES_NAME: _SHAPE, _OTHER_NAME: other_shape}


def test_stored_arrays_leaves_out_non_numeric_entries(tmp_path: Path) -> None:
    path = tmp_path / "mixed.mat"
    savemat(
        file_name=path,
        mdict={"values": np.ones((2, 2)), "label": "text", "info": {"a": 1}},
    )
    assert stored_arrays(path=path) == {"values": (2, 2)}
    text_only = tmp_path / "text.mat"
    savemat(file_name=text_only, mdict={"label": "text"})
    with pytest.raises(ValueError, match="no numeric array"):
        stored_arrays(path=text_only)


def test_stored_arrays_rejects_a_single_array_file(
    tmp_path: Path, lines: np.ndarray
) -> None:
    path = _save(path=tmp_path / "lines.npy", lines=lines)
    with pytest.raises(ValueError, match="single array"):
        stored_arrays(path=path)


@pytest.mark.parametrize("suffix", ARRAY_FILE_SUFFIXES)
def test_the_same_lines_read_alike_from_every_format(
    tmp_path: Path, lines: np.ndarray, suffix: str
) -> None:
    path = _save(path=tmp_path / f"lines{suffix}", lines=lines)
    name = _LINES_NAME if holds_several_arrays(path=path) else None
    values = load_stored_array(path=path, name=name)
    np.testing.assert_array_equal(values, lines)
    expected = select_radial_kspace(values=lines, layout=_LAYOUT)
    kspace = select_radial_kspace(values=values, layout=_LAYOUT)
    np.testing.assert_array_equal(kspace.values, expected.values)


# ======================================================================================
# Rejected names and files
# ======================================================================================


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_a_multi_array_file_needs_a_known_name(
    tmp_path: Path, lines: np.ndarray, suffix: str
) -> None:
    path = _save(path=tmp_path / f"lines{suffix}", lines=lines)
    with pytest.raises(ValueError, match="choose one by name"):
        load_stored_array(path=path)
    with pytest.raises(ValueError, match="no array named 'missing'"):
        load_stored_array(path=path, name="missing")
    with pytest.raises(TypeError):
        load_stored_array(path=path, name=1)  # type: ignore[arg-type]


def test_a_single_array_file_takes_no_name(tmp_path: Path, lines: np.ndarray) -> None:
    path = _save(path=tmp_path / "lines.npy", lines=lines)
    with pytest.raises(ValueError, match="without a name"):
        load_stored_array(path=path, name=_LINES_NAME)


@pytest.mark.parametrize("suffix", ARRAY_FILE_SUFFIXES)
@pytest.mark.parametrize("contents", [b"", b"short", b"not an array file at all" * 8])
def test_unreadable_files_are_value_errors(
    tmp_path: Path, suffix: str, contents: bytes
) -> None:
    path = tmp_path / f"broken{suffix}"
    path.write_bytes(data=contents)
    with pytest.raises(ValueError, match="cannot be read"):
        if holds_several_arrays(path=path):
            stored_arrays(path=path)
        else:
            load_stored_array(path=path)


def test_non_numeric_arrays_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "text.npy"
    np.save(file=path, arr=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="not a numeric array"):
        load_stored_array(path=path)
    mat = tmp_path / "text.mat"
    savemat(file_name=mat, mdict={"label": "text"})
    with pytest.raises(ValueError, match="not a numeric array"):
        load_stored_array(path=mat, name="label")


def test_invalid_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="suffix"):
        load_stored_array(path=tmp_path / "lines.txt")
    with pytest.raises(FileNotFoundError):
        load_stored_array(path=tmp_path / "missing.npy")
    with pytest.raises(TypeError):
        load_stored_array(path="lines.npy")  # type: ignore[arg-type]
