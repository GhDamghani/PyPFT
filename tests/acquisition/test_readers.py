"""Tests for ``pypft.acquisition``'s layout readers and their shared selection logic.

The fixture ``tests/samples/P01_slc1_phs1_lins196_cols416_cha4.mat`` is an excerpt of
the MD-CNN radial cine dataset in the dataset's own layout (one slice, one phase, 196
lines, a 416-sample readout, four coils; ``pypft.samples.Sample.MD_CNN_EXCERPT``),
stored as its ``data`` variable. Small synthetic arrays in that layout pin the
selection: which axes it chooses from, single samples against batches, the batch's
flattening order, and every rejected selection, array and layout. A layout reads
arrays, never files, so nothing here but the fixture touches the disk.
"""

from pathlib import Path

import numpy as np
import pytest

from pypft.acquisition import (
    RadialLineLayout,
    RadialLineLayoutError,
    load_stored_array,
    radial_lines_to_polar,
    select_radial_kspace,
    selectable_axes,
)

#: The downloaded excerpt of the MD-CNN dataset, and the name of its array.
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)
_FIXTURE_ARRAY = "data"

_LAYOUT = RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX

#: The synthetic array's ``(slices, phases, readout, lines, coils)`` sizes.
_SLICES, _PHASES, _READOUT, _LINES, _COILS = 2, 3, 16, 4, 5


@pytest.fixture(name="synthetic")
def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    """Build a small array in the dataset's layout, every sample distinct.

    :returns: The stored real array, ``(slices, phases, readout, lines, coils, 2)``,
        and its complex ``(slices, phases, readout, lines, coils)`` lines.
    :rtype: tuple[np.ndarray, np.ndarray]

    """
    shape = (_SLICES, _PHASES, _READOUT, _LINES, _COILS)
    stored = np.random.default_rng(seed=0).standard_normal(size=shape + (2,))
    return stored, stored[..., 0] + 1j * stored[..., 1]


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
# The downloaded excerpt
# ======================================================================================


def test_fixture_axes_and_layout() -> None:
    values = load_stored_array(path=_FIXTURE, name=_FIXTURE_ARRAY)
    assert selectable_axes(values=values, layout=_LAYOUT) == {
        "slice": 1,
        "phase": 1,
        "coil": 4,
    }
    kspace = select_radial_kspace(
        values=values, layout=_LAYOUT, selection={"coil": None}
    )
    # 196 lines become 392 spokes; the 416-sample readout, centered at 208, keeps
    # 208 samples per spoke, up to the readout's own Nyquist frequency.
    assert kspace.values.shape == (208, 392, 4)
    assert kspace.radius == pytest.approx(np.pi)


def test_fixture_spokes_share_their_line_center() -> None:
    values = load_stored_array(path=_FIXTURE, name=_FIXTURE_ARRAY)
    polar = select_radial_kspace(values=values, layout=_LAYOUT).values
    # Both halves of every line start at its center sample: the radius-0 row holds
    # each line's center twice, at its two opposite spokes.
    np.testing.assert_array_equal(polar[0, :196], polar[0, 196:])


# ======================================================================================
# Selection
# ======================================================================================


def test_selectable_axes_follow_the_shape(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, _ = synthetic
    axes = selectable_axes(values=stored, layout=_LAYOUT)
    assert axes == {"slice": _SLICES, "phase": _PHASES, "coil": _COILS}
    assert list(axes) == ["slice", "phase", "coil"]


def test_a_full_selection_reads_a_single_sample(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, lines = synthetic
    kspace = select_radial_kspace(
        values=stored,
        layout=_LAYOUT,
        selection={"slice": 1, "phase": 2, "coil": 3},
    )
    np.testing.assert_array_equal(kspace.values, _expected(lines[1, 2, :, :, 3]))


def test_omitted_axes_take_index_zero(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, lines = synthetic
    kspace = select_radial_kspace(values=stored, layout=_LAYOUT)
    np.testing.assert_array_equal(kspace.values, _expected(lines[0, 0, :, :, 0]))


def test_none_keeps_an_axis_in_the_batch(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, lines = synthetic
    kspace = select_radial_kspace(
        values=stored, layout=_LAYOUT, selection={"phase": 1, "coil": None}
    )
    assert kspace.values.shape == (_READOUT // 2, 2 * _LINES, _COILS)
    np.testing.assert_array_equal(kspace.values, _expected(lines[0, 1]))


def test_several_kept_axes_flatten_in_c_order(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, lines = synthetic
    kspace = select_radial_kspace(
        values=stored, layout=_LAYOUT, selection={"phase": None, "coil": None}
    )
    n_radial, n_angular = _READOUT // 2, 2 * _LINES
    assert kspace.values.shape == (n_radial, n_angular, _PHASES * _COILS)
    # The last kept axis varies fastest, so a reshape separates phase and coil.
    unflattened = kspace.values.reshape((n_radial, n_angular, _PHASES, _COILS))
    for phase in range(_PHASES):
        for coil in range(_COILS):
            np.testing.assert_array_equal(
                unflattened[:, :, phase, coil],
                _expected(lines[0, phase, :, :, coil]),
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
    synthetic: tuple[np.ndarray, np.ndarray],
    selection: object,
    error: type,
    match: str,
) -> None:
    stored, _ = synthetic
    with pytest.raises(error, match=match):
        select_radial_kspace(
            values=stored,
            layout=_LAYOUT,
            selection=selection,  # type: ignore[arg-type]
        )


# ======================================================================================
# Arrays and layouts
# ======================================================================================


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 8, 4, 2), (1, 1, 8, 4, 2, 3), (1, 1, 8, 4, 2, 2, 1)],
)
def test_arrays_in_another_layout_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(RadialLineLayoutError, match="must be"):
        selectable_axes(values=np.ones(shape=shape), layout=_LAYOUT)
    with pytest.raises(RadialLineLayoutError, match="must be"):
        select_radial_kspace(values=np.ones(shape=shape), layout=_LAYOUT)
    assert issubclass(RadialLineLayoutError, ValueError)


def test_a_complex_array_is_not_in_the_layout(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, _ = synthetic
    # The layout stores the real and imaginary parts on its last axis.
    with pytest.raises(RadialLineLayoutError, match="must be real"):
        select_radial_kspace(values=stored.astype(complex), layout=_LAYOUT)


def test_arrays_and_layouts_are_validated(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    stored, _ = synthetic
    with pytest.raises(TypeError):
        selectable_axes(values=stored.tolist(), layout=_LAYOUT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        select_radial_kspace(values=stored, layout=_LAYOUT.name)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        selectable_axes(values=stored, layout=1)  # type: ignore[arg-type]
