"""Tests for ``pypft.cli._io``, the terminal interface's file handling.

Loading is checked on every supported suffix, including the committed maze fixture,
a saved ``RadialKSpace`` and the downloaded MD-CNN acquisition excerpt; naming on single samples, batches and the batch index's
width growing past three digits; saving on the exact files written, their being valid
PNG images, the overwrite guard, the slowness warning, and every figure being closed.
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pypft.acquisition import (
    RadialKSpace,
    RadialLineLayout,
    RadialLineLayoutError,
    load_stored_array,
    select_radial_kspace,
)
from pypft.cli import _io
from pypft.cli._io import (
    LINE_FILE_SUFFIXES,
    POLAR_FILE_SUFFIXES,
    SLOW_BATCH_SIZE,
    SUPPORTED_SUFFIXES,
    CLIPerformanceWarning,
    FigurePart,
    acquisition_input,
    draw_signal,
    figure_paths,
    interactive_backend,
    load_array,
    output_filename,
    record_warnings,
    sample_of,
    save_signal_figures,
    warn_if_slow,
)
from pypft.domains import PolarDomain, PolarFrequencyAngularSignal
from pypft.grid import PolarGrid

#: The committed maze fixture: a (576, 48) 8-bit TIFF on its PolarGrid.
_MAZE = Path(__file__).resolve().parents[1] / "samples" / "maze_polar.tif"

#: The downloaded MD-CNN excerpt: one slice and phase, 4 coils of 196 lines.
_MD_CNN = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)
_MD_CNN_LAYOUT = RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX
_MD_CNN_ARRAY = "data"

#: A small grid, cheap to build, for the saving tests.
_GRID = PolarGrid(n_radial=12, n_angular=5, R=1.0)


def _signal(batch: int | None = None) -> PolarFrequencyAngularSignal:
    """A deterministic complex signal on ``_GRID``, a single sample or a batch."""
    shape = (_GRID.n_radial, _GRID.n_angular) + (() if batch is None else (batch,))
    rng = np.random.default_rng(seed=0)
    values = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return PolarFrequencyAngularSignal(values=values, grid=_GRID)


# ======================================================================================
# Loading
# ======================================================================================


def test_load_array_reads_the_maze_fixture_as_one_real_sample() -> None:
    loaded = load_array(path=_MAZE, batched=False)
    assert loaded.values.shape == (576, 48)
    assert loaded.values.dtype == np.float64
    assert loaded.is_image and loaded.radius is None
    np.testing.assert_array_equal(loaded.values, plt.imread(fname=_MAZE))


@pytest.mark.parametrize("batched", [False, True])
def test_load_array_reads_npy_samples_and_batches(
    tmp_path: Path, batched: bool
) -> None:
    shape = (6, 4, 3) if batched else (6, 4)
    values = np.arange(np.prod(shape)).reshape(shape) * (1 + 2j)
    path = tmp_path / "data.npy"
    np.save(file=path, arr=values)
    loaded = load_array(path=path, batched=batched)
    np.testing.assert_array_equal(loaded.values, values)
    assert not loaded.is_image and loaded.radius is None


def test_load_array_reads_radial_kspace_with_its_radius(tmp_path: Path) -> None:
    path = tmp_path / "kspace.npz"
    kspace = RadialKSpace(values=np.ones(shape=(8, 6), dtype=complex), radius=2.5)
    kspace.save(path=path)
    loaded = load_array(path=path, batched=False)
    assert loaded.values.shape == (8, 6)
    assert loaded.radius == 2.5 and not loaded.is_image


def test_load_array_reads_a_grayscale_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    # pyplot's imsave writes RGBA; Pillow writes the 8-bit grayscale ("L") image.
    from PIL import Image

    pixels = (np.arange(48).reshape(8, 6) * 5).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    loaded = load_array(path=path, batched=False)
    assert loaded.values.shape == (8, 6) and loaded.is_image
    np.testing.assert_allclose(loaded.values, pixels / 255.0)


def test_load_array_rejects_a_color_image(tmp_path: Path) -> None:
    path = tmp_path / "color.png"
    plt.imsave(fname=path, arr=np.eye(N=8), cmap="gray")
    with pytest.raises(ValueError, match="grayscale"):
        load_array(path=path, batched=False)


@pytest.mark.parametrize(
    ("shape", "batched"), [((6, 4), True), ((6, 4, 2), False), ((6,), False)]
)
def test_load_array_rejects_a_rank_that_does_not_match_the_choice(
    tmp_path: Path, shape: tuple[int, ...], batched: bool
) -> None:
    path = tmp_path / "data.npy"
    np.save(file=path, arr=np.zeros(shape=shape))
    with pytest.raises(ValueError, match="expected"):
        load_array(path=path, batched=batched)


def test_load_array_rejects_a_batch_of_images() -> None:
    with pytest.raises(ValueError, match="single sample"):
        load_array(path=_MAZE, batched=True)


def test_load_array_rejects_unsupported_suffixes(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text(data="1 2 3")
    with pytest.raises(ValueError, match="suffixes"):
        load_array(path=path, batched=False)


def test_load_array_rejects_non_numeric_and_non_finite_arrays(
    tmp_path: Path,
) -> None:
    text = tmp_path / "text.npy"
    np.save(file=text, arr=np.array(object=[["a", "b"], ["c", "d"]]))
    with pytest.raises(ValueError, match="not numbers"):
        load_array(path=text, batched=False)
    nan = tmp_path / "nan.npy"
    np.save(file=nan, arr=np.full(shape=(3, 3), fill_value=np.nan))
    with pytest.raises(ValueError, match="finite"):
        load_array(path=nan, batched=False)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.npy", b""),
        ("empty.npz", b""),
        ("broken.npz", b"PK\x03\x04 not an archive"),
        ("empty.png", b""),
        ("broken.png", b"\x89PNG\r\n\x1a\n not an image"),
    ],
)
def test_load_array_reports_an_empty_or_corrupt_file(
    tmp_path: Path, name: str, content: bytes
) -> None:
    path = tmp_path / name
    path.write_bytes(data=content)
    # OSError covers Pillow's UnidentifiedImageError; both are shown on screen.
    with pytest.raises((ValueError, OSError)):
        load_array(path=path, batched=False)


def test_load_array_rejects_an_archive_saved_as_npy(tmp_path: Path) -> None:
    path = tmp_path / "archive.npy"
    with path.open(mode="wb") as file:
        np.savez(file, values=np.ones(shape=(3, 3)))
    with pytest.raises(ValueError, match="archive"):
        load_array(path=path, batched=False)


def test_load_array_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_array(path=tmp_path / "missing.npy", batched=False)


def test_load_array_validates_argument_types() -> None:
    with pytest.raises(TypeError, match="Path"):
        load_array(path=str(_MAZE), batched=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="batched"):
        load_array(path=_MAZE, batched=1)  # type: ignore[arg-type]


# ======================================================================================
# Acquisitions
# ======================================================================================


def test_line_files_are_supported_but_a_mat_file_is_not_read_by_load_array() -> None:
    assert SUPPORTED_SUFFIXES == (*POLAR_FILE_SUFFIXES, ".mat")
    assert LINE_FILE_SUFFIXES == (".npy", ".npz", ".mat")
    with pytest.raises(ValueError, match="radial k-space lines"):
        load_array(path=_MD_CNN, batched=False)


@pytest.mark.parametrize(
    ("selection", "shape"),
    [({"coil": 2}, (208, 392)), ({"coil": None}, (208, 392, 4))],
)
def test_acquisition_input_converts_the_md_cnn_excerpt(
    selection: dict[str, int | None], shape: tuple[int, ...]
) -> None:
    full = {"slice": 0, "phase": 0, **selection}
    values = load_stored_array(path=_MD_CNN, name=_MD_CNN_ARRAY)
    loaded = acquisition_input(values=values, layout=_MD_CNN_LAYOUT, selection=full)
    expected = select_radial_kspace(
        values=values, layout=_MD_CNN_LAYOUT, selection=full
    )
    assert loaded.values.shape == shape
    np.testing.assert_array_equal(loaded.values, expected.values)
    assert loaded.radius == expected.radius
    assert not loaded.is_image


def test_acquisition_input_rejects_an_index_outside_its_axis() -> None:
    values = load_stored_array(path=_MD_CNN, name=_MD_CNN_ARRAY)
    with pytest.raises(ValueError, match="less than 4"):
        acquisition_input(values=values, layout=_MD_CNN_LAYOUT, selection={"coil": 4})


def test_acquisition_input_reports_an_array_in_another_layout() -> None:
    with pytest.raises(RadialLineLayoutError):
        acquisition_input(
            values=np.zeros(shape=(3, 4)), layout=_MD_CNN_LAYOUT, selection={}
        )


# ======================================================================================
# Naming
# ======================================================================================


def test_output_filename_for_a_single_sample() -> None:
    name = output_filename(
        stem="maze_polar",
        domain=PolarDomain.FREQUENCY_ANGULAR,
        part=FigurePart.MAGNITUDE,
    )
    assert name == "maze_polar_polar_frequency_angular_magnitude.png"


@pytest.mark.parametrize(
    ("batch_index", "batch_size", "suffix"),
    [
        (0, 2, "_b000"),
        (7, 10, "_b007"),
        (999, 1000, "_b999"),
        (0, 1001, "_b0000"),
        (1000, 1001, "_b1000"),
        (42, 12345, "_b00042"),
    ],
)
def test_output_filename_pads_the_batch_index_to_the_batch(
    batch_index: int, batch_size: int, suffix: str
) -> None:
    name = output_filename(
        stem="scan",
        domain=PolarDomain.SPATIAL_ANGULAR,
        part=FigurePart.PHASE,
        batch_index=batch_index,
        batch_size=batch_size,
    )
    assert name == f"scan_polar_spatial_angular_phase{suffix}.png"


@pytest.mark.parametrize(
    ("batch_index", "batch_size"), [(0, None), (None, 3), (3, 3), (-1, 3)]
)
def test_output_filename_rejects_an_invalid_batch_index(
    batch_index: int | None, batch_size: int | None
) -> None:
    with pytest.raises(ValueError):
        output_filename(
            stem="scan",
            domain=PolarDomain.SPATIAL_ANGULAR,
            part=FigurePart.PHASE,
            batch_index=batch_index,
            batch_size=batch_size,
        )


def test_output_filename_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="empty"):
        output_filename(
            stem="", domain=PolarDomain.SPATIAL_ANGULAR, part=FigurePart.PHASE
        )
    with pytest.raises(TypeError, match="Enum"):
        output_filename(
            stem="scan",
            domain=PolarDomain.SPATIAL_ANGULAR,
            part="phase",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="FigurePart"):
        output_filename(
            stem="scan",
            domain=PolarDomain.SPATIAL_ANGULAR,
            part=PolarDomain.SPATIAL_ANGULAR,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="str"):
        output_filename(
            stem=1,  # type: ignore[arg-type]
            domain=PolarDomain.SPATIAL_ANGULAR,
            part=FigurePart.PHASE,
        )


def test_sample_of_selects_one_element() -> None:
    signal = _signal(batch=3)
    sample = sample_of(signal=signal, batch_index=2)
    assert isinstance(sample, PolarFrequencyAngularSignal)
    np.testing.assert_array_equal(sample.values, signal.values[..., 2])
    single = _signal()
    assert sample_of(signal=single, batch_index=None) is single
    with pytest.raises(ValueError, match="no batch index"):
        sample_of(signal=single, batch_index=0)
    with pytest.raises(ValueError, match="needs a batch index"):
        sample_of(signal=signal, batch_index=None)
    with pytest.raises(ValueError, match="less than"):
        sample_of(signal=signal, batch_index=3)


# ======================================================================================
# Rendering and saving
# ======================================================================================


def test_interactive_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # The suite runs on Agg, which never opens a window.
    assert not interactive_backend()
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "TkAgg")
    assert interactive_backend()
    monkeypatch.setattr(
        matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline"
    )
    assert not interactive_backend()


def test_draw_signal_redraws_one_figure_in_place() -> None:
    figure = Figure()
    batch = _signal(batch=2)
    first = draw_signal(figure=figure, signal=sample_of(signal=batch, batch_index=0))
    second = draw_signal(figure=figure, signal=sample_of(signal=batch, batch_index=1))
    assert len(figure.axes) == 2
    assert first[0] is not second[0]
    with pytest.raises(TypeError, match="Figure"):
        draw_signal(figure=None, signal=_signal())  # type: ignore[arg-type]


def test_save_signal_figures_for_a_single_sample(tmp_path: Path) -> None:
    before = len(plt.get_fignums())
    directory = tmp_path / "figures" / "nested"
    saved = save_signal_figures(signal=_signal(), directory=directory, stem="scan")
    assert [path.name for path in saved] == [
        "scan_polar_frequency_angular_magnitude.png",
        "scan_polar_frequency_angular_phase.png",
    ]
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        path.name for path in saved
    )
    for path in saved:
        assert plt.imread(fname=path).ndim == 3
    assert len(plt.get_fignums()) == before


def test_save_signal_figures_for_a_batch(tmp_path: Path) -> None:
    before = len(plt.get_fignums())
    batch = _signal(batch=3)
    saved = save_signal_figures(signal=batch, directory=tmp_path, stem="b")
    assert [path.name for path in saved] == [
        f"b_polar_frequency_angular_{part}_b{index:03d}.png"
        for index in range(3)
        for part in ("magnitude", "phase")
    ]
    assert saved == figure_paths(signal=batch, directory=tmp_path, stem="b")
    for path in saved:
        assert plt.imread(fname=path).ndim == 3
    assert len(plt.get_fignums()) == before


def test_save_signal_figures_never_overwrites_silently(tmp_path: Path) -> None:
    saved = save_signal_figures(signal=_signal(), directory=tmp_path, stem="scan")
    saved[0].unlink()
    with pytest.raises(FileExistsError, match="exist"):
        save_signal_figures(signal=_signal(), directory=tmp_path, stem="scan")
    # Nothing was written: the removed file is still missing.
    assert not saved[0].exists()
    save_signal_figures(
        signal=_signal(), directory=tmp_path, stem="scan", overwrite=True
    )
    assert all(path.exists() for path in saved)


def test_slowness_warning_fires_above_the_threshold_only() -> None:
    with pytest.warns(CLIPerformanceWarning, match="may be slow"):
        warn_if_slow(batch_size=SLOW_BATCH_SIZE + 1)
    with warnings.catch_warnings():
        warnings.simplefilter(action="error")
        warn_if_slow(batch_size=SLOW_BATCH_SIZE)
        warn_if_slow(batch_size=None)


def test_saving_a_large_batch_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_io, "SLOW_BATCH_SIZE", 1)
    with pytest.warns(CLIPerformanceWarning):
        save_signal_figures(signal=_signal(batch=2), directory=tmp_path, stem="b")


def test_record_warnings_keeps_warnings_from_escaping() -> None:
    def noisy() -> int:
        warnings.warn("first", UserWarning, stacklevel=1)
        warn_if_slow(batch_size=SLOW_BATCH_SIZE + 1)
        return 7

    result, messages = record_warnings(call=noisy)
    assert result == 7
    assert messages[0] == "UserWarning: first"
    assert messages[1].startswith("CLIPerformanceWarning: ")
