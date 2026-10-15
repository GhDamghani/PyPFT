"""Tests for ``pypft.acquisition``'s layout readers and their shared selection logic.

The committed fixture ``tests/samples/P01_slc1_phs1_lins196_cols416_cha4.mat`` is an
excerpt of the MD-CNN radial cine dataset in the dataset's own layout (one slice, one
phase, 196 lines, a 416-sample readout, four coils; ``scripts/make_md_cnn_excerpt.py``).
Small synthetic files in that layout pin the selection: which axes it chooses from,
single samples against batches, the batch's flattening order, and every rejected
selection, path, file and layout.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pypft.acquisition import (
    RadialLineLayout,
    RadialLineLayoutError,
    layout_suffix,
    radial_lines_to_polar,
    read_radial_kspace,
    selectable_axes,
)

#: The committed excerpt of the MD-CNN dataset.
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)

_LAYOUT = RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX

#: The synthetic file's ``(slices, phases, readout, lines, coils)`` sizes.
_SLICES, _PHASES, _READOUT, _LINES, _COILS = 2, 3, 16, 4, 5


@pytest.fixture(name="synthetic")
def _synthetic(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Write a small file in the dataset's layout, every sample distinct.

    :param tmp_path: pytest's temporary directory.
    :type tmp_path: Path
    :returns: The file and its complex ``(slices, phases, readout, lines, coils)``
        data.
    :rtype: tuple[Path, np.ndarray]

    """
    shape = (_SLICES, _PHASES, _READOUT, _LINES, _COILS)
    data = np.random.default_rng(seed=0).standard_normal(size=shape + (2,))
    path = tmp_path / "P99_slc2_phs3_lins4_cols16_cha5.mat"
    savemat(file_name=path, mdict={"data": data})
    return path, data[..., 0] + 1j * data[..., 1]


def _expected(lines: np.ndarray) -> np.ndarray:
    """Convert ``(readout, line[, batch])`` lines with the dataset's geometry.

    :param lines: The lines.
    :type lines: np.ndarray
    :returns: The converted values.
    :rtype: np.ndarray

    """
    return radial_lines_to_polar(
        lines=lines,
        angles=np.arange(lines.shape[1]) * (np.pi / lines.shape[1]),
        center_index=lines.shape[0] // 2,
    ).values


# ======================================================================================
# The committed excerpt
# ======================================================================================


def test_fixture_axes_and_layout() -> None:
    assert selectable_axes(path=_FIXTURE, layout=_LAYOUT) == {
        "slice": 1,
        "phase": 1,
        "coil": 4,
    }
    kspace = read_radial_kspace(path=_FIXTURE, layout=_LAYOUT, selection={"coil": None})
    # 196 lines become 392 spokes; the 416-sample readout, centered at 208, keeps
    # 208 samples per spoke, up to the readout's own Nyquist frequency.
    assert kspace.values.shape == (208, 392, 4)
    assert kspace.radius == pytest.approx(np.pi)


def test_fixture_spokes_share_their_line_center() -> None:
    values = read_radial_kspace(path=_FIXTURE, layout=_LAYOUT).values
    # Both halves of every line start at its center sample: the radius-0 row holds
    # each line's center twice, at its two opposite spokes.
    np.testing.assert_array_equal(values[0, :196], values[0, 196:])


# ======================================================================================
# Selection
# ======================================================================================


def test_selectable_axes_reads_the_header(synthetic: tuple[Path, np.ndarray]) -> None:
    path, _ = synthetic
    axes = selectable_axes(path=path, layout=_LAYOUT)
    assert axes == {"slice": _SLICES, "phase": _PHASES, "coil": _COILS}
    assert list(axes) == ["slice", "phase", "coil"]


def test_a_full_selection_reads_a_single_sample(
    synthetic: tuple[Path, np.ndarray],
) -> None:
    path, data = synthetic
    kspace = read_radial_kspace(
        path=path,
        layout=_LAYOUT,
        selection={"slice": 1, "phase": 2, "coil": 3},
    )
    np.testing.assert_array_equal(kspace.values, _expected(data[1, 2, :, :, 3]))


def test_omitted_axes_take_index_zero(synthetic: tuple[Path, np.ndarray]) -> None:
    path, data = synthetic
    kspace = read_radial_kspace(path=path, layout=_LAYOUT)
    np.testing.assert_array_equal(kspace.values, _expected(data[0, 0, :, :, 0]))


def test_none_keeps_an_axis_in_the_batch(synthetic: tuple[Path, np.ndarray]) -> None:
    path, data = synthetic
    kspace = read_radial_kspace(
        path=path, layout=_LAYOUT, selection={"phase": 1, "coil": None}
    )
    expected = _expected(data[0, 1])
    assert kspace.values.shape == (_READOUT // 2, 2 * _LINES, _COILS)
    np.testing.assert_array_equal(kspace.values, expected)


def test_several_kept_axes_flatten_in_c_order(
    synthetic: tuple[Path, np.ndarray],
) -> None:
    path, data = synthetic
    kspace = read_radial_kspace(
        path=path, layout=_LAYOUT, selection={"phase": None, "coil": None}
    )
    n_radial, n_angular = _READOUT // 2, 2 * _LINES
    assert kspace.values.shape == (n_radial, n_angular, _PHASES * _COILS)
    # The last kept axis varies fastest, so a reshape separates phase and coil.
    unflattened = kspace.values.reshape((n_radial, n_angular, _PHASES, _COILS))
    for phase in range(_PHASES):
        for coil in range(_COILS):
            np.testing.assert_array_equal(
                unflattened[:, :, phase, coil],
                _expected(data[0, phase, :, :, coil]),
            )


@pytest.mark.parametrize(
    ("selection", "error", "match"),
    [
        ({"echo": 0}, ValueError, "unknown axis"),
        ({"coil": _COILS}, ValueError, "less than"),
        ({"coil": -1}, ValueError, "non-negative"),
        ({"coil": 1.0}, TypeError, "int"),
        ({0: 1}, TypeError, "str"),
        ([("coil", 0)], TypeError, "Mapping"),
        ([], TypeError, "Mapping"),
    ],
)
def test_invalid_selections_are_rejected(
    synthetic: tuple[Path, np.ndarray], selection: object, error: type, match: str
) -> None:
    path, _ = synthetic
    with pytest.raises(error, match=match):
        read_radial_kspace(
            path=path,
            layout=_LAYOUT,
            selection=selection,  # type: ignore[arg-type]
        )


# ======================================================================================
# Paths, files and layouts
# ======================================================================================


def test_files_in_another_layout_are_rejected(tmp_path: Path) -> None:
    no_data = tmp_path / "other.mat"
    savemat(file_name=no_data, mdict={"kspace": np.ones((2, 2))})
    with pytest.raises(RadialLineLayoutError, match="no 'data'"):
        selectable_axes(path=no_data, layout=_LAYOUT)
    wrong_rank = tmp_path / "rank.mat"
    savemat(file_name=wrong_rank, mdict={"data": np.ones((1, 1, 8, 4, 2))})
    with pytest.raises(RadialLineLayoutError, match="must be"):
        read_radial_kspace(path=wrong_rank, layout=_LAYOUT)
    assert issubclass(RadialLineLayoutError, ValueError)


@pytest.mark.parametrize("contents", [b"", b"short", b"not a MATLAB file at all" * 8])
def test_unreadable_files_are_layout_errors(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "broken.mat"
    path.write_bytes(data=contents)
    with pytest.raises(RadialLineLayoutError, match="not a readable MATLAB 5 file"):
        selectable_axes(path=path, layout=_LAYOUT)


def test_invalid_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="suffix"):
        selectable_axes(path=tmp_path / "data.npz", layout=_LAYOUT)
    with pytest.raises(FileNotFoundError):
        read_radial_kspace(path=tmp_path / "missing.mat", layout=_LAYOUT)
    with pytest.raises(TypeError):
        selectable_axes(path=str(_FIXTURE), layout=_LAYOUT)  # type: ignore[arg-type]


def test_layouts_are_validated() -> None:
    assert layout_suffix(layout=_LAYOUT) == ".mat"
    with pytest.raises(TypeError):
        layout_suffix(layout=_LAYOUT.name)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        selectable_axes(path=_FIXTURE, layout=1)  # type: ignore[arg-type]
