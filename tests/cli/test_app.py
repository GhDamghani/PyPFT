"""Tests for ``pypft.cli``'s app, the terminal interface, driven headlessly.

The app runs through Textual's ``App.run_test``, inside ``asyncio.run`` so no pytest
plugin is needed. Buttons are pressed through ``Button.press``, and every step waits
for the screen it leads to, or for the status line to change, rather than for a fixed
time. The suite runs on matplotlib's ``Agg`` backend, so viewing writes a file; the
window path is exercised by pretending the backend opens windows.
"""

import asyncio
import subprocess
import sys
import webbrowser
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Select, Static
from textual_fspicker.base_dialog import Dialog
from textual_fspicker.parts import DirectoryNavigation
from textual_fspicker.parts.directory_navigation import DirectoryEntry

from pypft.acquisition import (
    RadialKSpace,
    RadialLineLayout,
    load_stored_array,
    select_radial_kspace,
)
from pypft.cli import TUTORIALS_URL, PyPFTApp, _app, _io, main
from pypft.cli._app import (
    AcquisitionScreen,
    ArrayScreen,
    ConfirmScreen,
    DirectionScreen,
    FilePickerScreen,
    FileScreen,
    GridScreen,
    InputLayout,
    ResultScreen,
    SampleLayout,
    SampleScreen,
    StartScreen,
    StepScreen,
    TutorialsScreen,
    input_suffixes,
    is_input_file,
)
from pypft.cli._pipeline import InputGrid, Route
from pypft.grid import (
    PolarGrid,
    resample_uniform_polar,
    resample_uniform_polar_frequency,
)
from pypft.transform import Direction, forward_pft, inverse_pft

#: The committed maze fixture: a (576, 48) 8-bit TIFF on PolarGrid(576, 48, R=1).
_MAZE = Path(__file__).resolve().parents[1] / "samples" / "maze_polar.tif"

#: The downloaded MD-CNN excerpt: one slice and phase, 4 coils of 196 lines.
_MD_CNN = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)

#: The MD-CNN excerpt's array, and its layout.
_MD_CNN_ARRAY = "data"
_LINE_LAYOUT = RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX

#: A terminal tall enough to show every widget at once.
_SIZE = (110, 70)

#: How long a step may take, and how often that is checked, in seconds; the maze's
#: first forward transform builds its kernels, several seconds.
_STEP_TIMEOUT, _STEP_POLL = 60.0, 0.05

#: A small grid for the tests that need no particular file.
_N_RADIAL, _N_ANGULAR, _R = 16, 5, 4.0
_GRID = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)


# ======================================================================================
# Driving the app
# ======================================================================================


def _drive(steps: Callable[[Pilot], Awaitable[None]]) -> PyPFTApp:
    """Run a fresh app headlessly, drive it with ``steps``, and return it.

    :param steps: What to do with the running app's pilot.
    :type steps: Callable[[Pilot], Awaitable[None]]
    :returns: The app, after it stopped.
    :rtype: PyPFTApp

    """
    app = PyPFTApp()

    async def run() -> None:
        async with app.run_test(size=_SIZE) as pilot:
            await _wait(pilot=pilot, done=lambda: isinstance(app.screen, StartScreen))
            await steps(pilot)

    asyncio.run(run())
    return app


async def _wait(pilot: Pilot, done: Callable[[], bool]) -> None:
    """Wait until ``done`` holds.

    :raises AssertionError: If it does not hold within ``_STEP_TIMEOUT``.

    """
    for _ in range(int(_STEP_TIMEOUT / _STEP_POLL)):
        if done():
            return
        await pilot.pause(delay=_STEP_POLL)
    raise AssertionError("the app did not reach the expected state")


def _status(pilot: Pilot) -> str:
    """The current screen's status line."""
    app = pilot.app
    assert isinstance(app, PyPFTApp)
    return app.status


def _screen(pilot: Pilot) -> Screen:
    """The app's current screen."""
    return pilot.app.screen


async def _press(pilot: Pilot, button_id: str) -> None:
    """Press a button and wait until the screen or its status line changes."""
    app = pilot.app
    assert isinstance(app, PyPFTApp)
    screen, status = app.screen, app.status
    app.screen.query_one(f"#{button_id}", Button).press()
    # A long step first shows a status ending in "..." and replaces it when done.
    await _wait(
        pilot=pilot,
        done=lambda: (app.screen is not screen or app.status != status)
        and not app.status.endswith("..."),
    )


async def _quit(pilot: Pilot) -> None:
    """Press Quit and wait until the app has its exit code."""
    pilot.app.screen.query_one("#quit", Button).press()
    await _wait(pilot=pilot, done=lambda: pilot.app.return_value is not None)


def _set(pilot: Pilot, **fields: str) -> None:
    """Type into the current screen's text fields, by id (``_`` for ``-``)."""
    for field_id, text in fields.items():
        selector = f"#{field_id.replace('_', '-')}"
        _screen(pilot).query_one(selector, Input).value = text


def _select(pilot: Pilot, select_id: str, value: object) -> None:
    """Choose an option of one of the current screen's selectors."""
    _screen(pilot).query_one(f"#{select_id}", Select).value = value


async def _to_file(pilot: Pilot, layout: InputLayout = SampleLayout.SINGLE) -> None:
    """Go from the start screen to the file screen, choosing ``layout``."""
    await _press(pilot=pilot, button_id="transform")
    assert isinstance(_screen(pilot), SampleScreen)
    _select(pilot=pilot, select_id="layout", value=layout)
    await _press(pilot=pilot, button_id="next")
    assert isinstance(_screen(pilot), FileScreen)


async def _load(pilot: Pilot, path: Path) -> None:
    """Type a file into the file screen and load it."""
    _set(pilot=pilot, path=str(path))
    await _press(pilot=pilot, button_id="load")


async def _load_lines(pilot: Pilot, path: Path, array: str | None = None) -> None:
    """Load a file of radial lines, choosing ``array`` on the array screen if any."""
    await _load(pilot=pilot, path=path)
    if array is not None:
        assert isinstance(_screen(pilot), ArrayScreen), _status(pilot=pilot)
        _select(pilot=pilot, select_id="array", value=array)
        await _press(pilot=pilot, button_id="next")


async def _to_direction(
    pilot: Pilot,
    path: Path,
    R: float,
    layout: SampleLayout = SampleLayout.SINGLE,
    radius: float | None = None,
) -> None:
    """Go from the start screen to the direction screen with one file and grid.

    With ``radius``, the file is described as uniform polar data covering it.
    """
    await _to_file(pilot=pilot, layout=layout)
    await _load(pilot=pilot, path=path)
    assert isinstance(_screen(pilot), GridScreen), pilot.app.screen
    _set(pilot=pilot, R=repr(R))
    if radius is not None:
        _select(pilot=pilot, select_id="input-grid", value=InputGrid.UNIFORM_POLAR)
        await pilot.pause()
        _set(pilot=pilot, radius=repr(radius))
    await _press(pilot=pilot, button_id="next")
    assert isinstance(_screen(pilot), DirectionScreen)


def _save_npy(path: Path, values: np.ndarray) -> Path:
    """Write ``values`` to ``path`` and return it."""
    np.save(file=path, arr=values)
    return path


def _batch_file(tmp_path: Path, size: int) -> Path:
    """A batch of ``size`` Gaussians on ``_GRID``, as a ``.npy`` file."""
    sample = np.exp(-(_GRID.r.T**2))
    batch = np.stack(arrays=[(k + 1) * sample for k in range(size)], axis=-1)
    return _save_npy(path=tmp_path / "batch.npy", values=batch)


# ======================================================================================
# Navigation
# ======================================================================================


def test_the_maze_reaches_the_result_and_back_returns_step_by_step(
    tmp_path: Path,
) -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=_MAZE, R=1.0)
        await _press(pilot=pilot, button_id="run")
        assert isinstance(_screen(pilot), ResultScreen)
        # The result is forward_pft of the fixture on its own grid.
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        session = screen.session
        assert session.result is not None
        grid = PolarGrid(n_radial=576, n_angular=48, R=1.0)
        expected = forward_pft(f=plt.imread(fname=_MAZE).astype(np.float64), grid=grid)
        np.testing.assert_array_equal(session.result.signals[-1].values, expected)
        # A headless save writes exactly the two figures of the single result.
        _set(pilot=pilot, directory=str(tmp_path))
        await _press(pilot=pilot, button_id="save")
        await _wait(pilot=pilot, done=lambda: "Saved" in _status(pilot=pilot))
        names = sorted(path.name for path in tmp_path.iterdir())
        assert names == [
            "maze_polar_polar_frequency_angular_magnitude.png",
            "maze_polar_polar_frequency_angular_phase.png",
        ]
        for name in names:
            assert plt.imread(fname=tmp_path / name).ndim == 3
        for expected_screen in (DirectionScreen, GridScreen, FileScreen, SampleScreen):
            await _press(pilot=pilot, button_id="back")
            assert isinstance(_screen(pilot), expected_screen)
        await _quit(pilot=pilot)

    assert _drive(steps=steps).return_value == 0


def test_the_start_screen_has_no_back_and_quit_exits_with_zero() -> None:
    async def steps(pilot: Pilot) -> None:
        assert not _screen(pilot).query("#back")
        await _quit(pilot=pilot)

    assert _drive(steps=steps).return_value == 0


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv=["--help"])
    assert exit_info.value.code == 0
    assert "pypft" in capsys.readouterr().out
    # The module entry point, in its own interpreter.
    completed = subprocess.run(
        args=[sys.executable, "-m", "pypft", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage: pypft" in completed.stdout


def test_the_file_picker_fills_the_file_field(tmp_path: Path) -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        _set(pilot=pilot, path=str(_MAZE))
        await _press(pilot=pilot, button_id="browse")
        picker = _screen(pilot)
        assert isinstance(picker, FilePickerScreen)
        picker.dismiss(result=_MAZE)
        await _wait(pilot=pilot, done=lambda: isinstance(_screen(pilot), FileScreen))
        await _wait(pilot=pilot, done=lambda: "Picked" in _status(pilot=pilot))
        field = _screen(pilot).query_one("#path", Input)
        assert field.value == str(_MAZE)
        # Cancel leaves the field as it was.
        await _press(pilot=pilot, button_id="browse")
        await _press(pilot=pilot, button_id="cancel")
        assert field.value == str(_MAZE)

    _drive(steps=steps)


async def _open_picker(pilot: Pilot, layout: InputLayout, start: Path) -> None:
    """Go to the file screen and open the file dialog in the directory ``start``."""
    await _to_file(pilot=pilot, layout=layout)
    _set(pilot=pilot, path=str(start / "file"))
    await _press(pilot=pilot, button_id="browse")
    assert isinstance(_screen(pilot), FilePickerScreen)


def _picker_listing(pilot: Pilot) -> list[str]:
    """The names the file dialog lists, its parent entry left out."""
    navigation = _screen(pilot).query_one(DirectoryNavigation)
    names = [
        cast(DirectoryEntry, navigation.get_option_at_index(index)).location.name
        for index in range(navigation.option_count)
    ]
    return sorted(name for name in names if name != "..")


def _type_into_picker(pilot: Pilot, text: str) -> None:
    """Type a path into the file dialog's field and press its Open button."""
    picker = _screen(pilot)
    picker.query_one(Input).value = text
    picker.query_one("#select", Button).press()


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (SampleLayout.SINGLE, ["a.npy", "b.TIF", "c.png", "e.npz", "folder"]),
        (_LINE_LAYOUT, ["a.npy", "e.npz", "f.mat", "folder"]),
    ],
)
def test_the_file_dialog_lists_folders_and_the_layouts_files_only(
    tmp_path: Path, layout: InputLayout, expected: list[str]
) -> None:
    names = ("a.npy", "b.TIF", "c.png", "d.txt", ".hidden.npy", "e.npz", "f.mat")
    for name in names:
        (tmp_path / name).write_bytes(data=b"")
    (tmp_path / "folder").mkdir()

    async def steps(pilot: Pilot) -> None:
        await _open_picker(pilot=pilot, layout=layout, start=tmp_path)
        await _wait(pilot=pilot, done=lambda: _picker_listing(pilot=pilot) == expected)

    _drive(steps=steps)


@pytest.mark.parametrize(
    ("name", "accepted"),
    [("a.npy", True), ("b.TIF", True), ("c.png", True), ("d.txt", False)],
)
def test_input_files_are_told_by_suffix_in_any_case(name: str, accepted: bool) -> None:
    suffixes = input_suffixes(layout=SampleLayout.SINGLE)
    assert is_input_file(path=Path(name), suffixes=suffixes) is accepted


def test_a_directory_typed_into_the_file_dialog_is_opened(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "folder"
    target.mkdir(parents=True)
    (target / "inside.npy").write_bytes(data=b"")

    async def steps(pilot: Pilot) -> None:
        await _open_picker(pilot=pilot, layout=SampleLayout.SINGLE, start=tmp_path)
        _type_into_picker(pilot=pilot, text=str(target))
        await _wait(
            pilot=pilot, done=lambda: _picker_listing(pilot=pilot) == ["inside.npy"]
        )
        navigation = _screen(pilot).query_one(DirectoryNavigation)
        assert navigation.location == target.resolve()
        assert isinstance(_screen(pilot), FilePickerScreen)

    _drive(steps=steps)


def test_a_file_typed_into_the_file_dialog_is_picked(tmp_path: Path) -> None:
    async def steps(pilot: Pilot) -> None:
        await _open_picker(pilot=pilot, layout=SampleLayout.SINGLE, start=tmp_path)
        _type_into_picker(pilot=pilot, text=str(_MAZE))
        await _wait(pilot=pilot, done=lambda: isinstance(_screen(pilot), FileScreen))
        assert _screen(pilot).query_one("#path", Input).value == str(_MAZE)

    _drive(steps=steps)


@pytest.mark.parametrize(
    ("name", "message"),
    [("notes.txt", "Not a "), ("missing.npy", "must exist")],
)
def test_the_file_dialog_refuses_other_files(
    tmp_path: Path, name: str, message: str
) -> None:
    (tmp_path / "notes.txt").write_bytes(data=b"")

    async def steps(pilot: Pilot) -> None:
        await _open_picker(pilot=pilot, layout=SampleLayout.SINGLE, start=tmp_path)
        _type_into_picker(pilot=pilot, text=str(tmp_path / name))
        dialog = _screen(pilot).query_one(Dialog)
        await _wait(pilot=pilot, done=lambda: message in str(dialog.border_subtitle))
        assert isinstance(_screen(pilot), FilePickerScreen)

    _drive(steps=steps)


def test_tutorials_open_the_online_docs_and_local_notebooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    started: list[list[str]] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(_app, "notebooks_directory", lambda: tmp_path)
    monkeypatch.setattr(
        subprocess, "Popen", lambda args, **kwargs: started.append(args)
    )

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="tutorials")
        assert isinstance(_screen(pilot), TutorialsScreen)
        await _press(pilot=pilot, button_id="open-url")
        await _press(pilot=pilot, button_id="jupyter")
        await _press(pilot=pilot, button_id="back")
        assert isinstance(_screen(pilot), StartScreen)

    _drive(steps=steps)
    assert opened == [TUTORIALS_URL]
    assert started == [[sys.executable, "-m", "jupyterlab", str(tmp_path)]]


# ======================================================================================
# Grids, warnings and paths
# ======================================================================================


def test_an_adequacy_warning_is_shown_on_screen_and_does_not_escape() -> None:
    # Under filterwarnings = ["error"], an escaping AdequacyWarning fails this test.
    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=_MAZE, R=1.0)
        screen = _screen(pilot)
        assert isinstance(screen, DirectionScreen)
        assert "AdequacyWarning: n_radial=576 is likely inadequate" in screen.notices

    _drive(steps=steps)


def test_uniform_input_goes_through_resample_uniform_polar(tmp_path: Path) -> None:
    r = np.arange(40) * _R / 40
    values = np.repeat(np.exp(-(r**2))[:, np.newaxis], repeats=_N_ANGULAR, axis=1)
    path = _save_npy(path=tmp_path / "uniform.npy", values=values)

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        await _load(pilot=pilot, path=path)
        _set(pilot=pilot, R=repr(_R), n_radial=str(_N_RADIAL), radius=repr(_R))
        _select(pilot=pilot, select_id="input-grid", value=InputGrid.UNIFORM_POLAR)
        await _press(pilot=pilot, button_id="next")
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        expected = forward_pft(
            f=resample_uniform_polar(values=values, grid=_GRID, radius=_R), grid=_GRID
        )
        result = screen.session.result
        assert result is not None
        np.testing.assert_array_equal(result.signals[-1].values, expected)
        assert any("interpolated" in notice for notice in result.notices)

    _drive(steps=steps)


def test_radial_kspace_is_uniform_ipft_input_with_its_radius(tmp_path: Path) -> None:
    radius = 1.5 * float(np.max(a=_GRID.rho))
    rho = np.arange(30) * radius / 30
    values = np.repeat(
        np.exp(-(rho**2) / 4)[:, np.newaxis], repeats=_N_ANGULAR, axis=1
    ).astype(complex)
    path = tmp_path / "kspace.npz"
    RadialKSpace(values=values, radius=radius).save(path=path)

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        await _load(pilot=pilot, path=path)
        grid_screen = _screen(pilot)
        assert isinstance(grid_screen, GridScreen)
        # The file's radius and uniform layout are filled in.
        assert grid_screen.query_one("#radius", Input).value == repr(radius)
        select = grid_screen.query_one("#input-grid", Select)
        # The selector takes its value once mounted, which may follow the screen.
        await _wait(pilot=pilot, done=lambda: select.value is not None)
        assert select.value is InputGrid.UNIFORM_POLAR
        _set(
            pilot=pilot, R=repr(_R), n_radial=str(_N_RADIAL), n_angular=str(_N_ANGULAR)
        )
        await _press(pilot=pilot, button_id="next")
        # k-space is frequency input: the IPFT is chosen.
        direction = _screen(pilot).query_one("#direction", Select).value
        assert direction is Direction.INVERSE
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        result = screen.session.result
        assert result is not None
        expected = inverse_pft(
            F=resample_uniform_polar_frequency(
                values=values, grid=_GRID, radius=radius
            ),
            grid=_GRID,
        )
        np.testing.assert_array_equal(result.signals[-1].values, expected)

    _drive(steps=steps)


def test_an_acquisition_file_is_selected_read_and_inverted() -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_file(
            pilot=pilot, layout=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX
        )
        await _load(pilot=pilot, path=_MD_CNN)
        # A .mat file holds named arrays: the one with the lines is chosen first.
        array_screen = _screen(pilot)
        assert isinstance(array_screen, ArrayScreen)
        assert array_screen.query_one("#array", Select).value == _MD_CNN_ARRAY
        await _press(pilot=pilot, button_id="next")
        screen = _screen(pilot)
        assert isinstance(screen, AcquisitionScreen)
        # Every axis starts at index 0: a single sample.
        fields = {field.id: field.value for field in screen.query(Input)}
        assert fields == {"axis-slice": "0", "axis-phase": "0", "axis-coil": "0"}
        # A field that is neither an index nor "all" is reported, not raised.
        _set(pilot=pilot, axis_coil="first")
        await _press(pilot=pilot, button_id="read")
        assert "Error: an axis takes an index or 'all'" in _status(pilot=pilot)
        _set(pilot=pilot, axis_coil="all")
        await _press(pilot=pilot, button_id="read")
        grid_screen = _screen(pilot)
        assert isinstance(grid_screen, GridScreen)
        # The k-space's grid is proposed: the readout's field of view, in pixels.
        assert grid_screen.query_one("#R", Input).value == "208.0"
        assert grid_screen.query_one("#n-angular", Input).value == "98"
        assert grid_screen.query_one("#n-radial", Input).value == "183"
        select = grid_screen.query_one("#input-grid", Select)
        # The selector takes its value once mounted, which may follow the screen.
        await _wait(pilot=pilot, done=lambda: select.value is not None)
        assert select.value is InputGrid.UNIFORM_POLAR
        # A small grid keeps the test short.
        _set(pilot=pilot, n_radial=str(_N_RADIAL), n_angular=str(_N_ANGULAR))
        await _press(pilot=pilot, button_id="next")
        direction = _screen(pilot).query_one("#direction", Select).value
        assert direction is Direction.INVERSE
        await _press(pilot=pilot, button_id="run")
        result_screen = _screen(pilot)
        assert isinstance(result_screen, ResultScreen)
        result = result_screen.session.result
        assert result is not None
        kspace = select_radial_kspace(
            values=load_stored_array(path=_MD_CNN, name=_MD_CNN_ARRAY),
            layout=_LINE_LAYOUT,
            selection={"coil": None},
        )
        grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=208.0)
        expected = inverse_pft(
            F=resample_uniform_polar_frequency(
                values=kspace.values, grid=grid, radius=kspace.radius
            ),
            grid=grid,
        )
        np.testing.assert_array_equal(result.signals[-1].values, expected)
        for expected_screen in (
            DirectionScreen,
            GridScreen,
            AcquisitionScreen,
            ArrayScreen,
            FileScreen,
        ):
            await _press(pilot=pilot, button_id="back")
            assert isinstance(_screen(pilot), expected_screen)

    _drive(steps=steps)


def test_an_acquisition_is_a_layout_choice() -> None:
    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="transform")
        # Every layout of radial lines is offered next to the two polar layouts.
        options = _app.layout_options()
        assert [value for _, value in options] == [
            *SampleLayout,
            *RadialLineLayout,
        ]
        _select(
            pilot=pilot,
            select_id="layout",
            value=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX,
        )
        await _press(pilot=pilot, button_id="next")
        assert isinstance(_screen(pilot), FileScreen)
        # A .mat file is read only as radial lines.
        await _press(pilot=pilot, button_id="back")
        _select(pilot=pilot, select_id="layout", value=SampleLayout.SINGLE)
        await _press(pilot=pilot, button_id="next")
        await _load(pilot=pilot, path=_MD_CNN)
        assert "radial k-space lines" in _status(pilot=pilot)

    _drive(steps=steps)


def test_the_layout_note_shows_only_each_radial_line_layouts_own_text() -> None:
    def note(pilot: Pilot) -> Static:
        return _screen(pilot).query_one("#layout-note", Static)

    def pick(pilot: Pilot, layout: InputLayout) -> None:
        # A pick from the list posts Changed; setting the value alone may not.
        select = _screen(pilot).query_one("#layout", Select)
        select.value = layout
        # The message's value parameter is typed by a type variable pyright cannot
        # solve outside the class.
        select.post_message(Select.Changed(select, cast(Any, layout)))

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="transform")
        assert not note(pilot).display
        for layout in RadialLineLayout:
            pick(pilot=pilot, layout=layout)
            await _wait(pilot=pilot, done=lambda: note(pilot).display)
            assert _app.layout_note(layout=layout)
        for layout in SampleLayout:
            pick(pilot=pilot, layout=layout)
            await _wait(pilot=pilot, done=lambda: not note(pilot).display)
            assert _app.layout_note(layout=layout) == ""

    _drive(steps=steps)


def test_a_single_sample_of_an_acquisition_takes_one_index_per_axis() -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_file(
            pilot=pilot, layout=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX
        )
        await _load_lines(pilot=pilot, path=_MD_CNN, array=_MD_CNN_ARRAY)
        assert isinstance(_screen(pilot), AcquisitionScreen)
        _set(pilot=pilot, axis_coil="3")
        await _press(pilot=pilot, button_id="read")
        screen = _screen(pilot)
        assert isinstance(screen, GridScreen)
        loaded = screen.session.loaded
        assert loaded is not None
        assert loaded.values.shape == (208, 392)

    _drive(steps=steps)


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_lines_read_alike_from_a_numpy_file(tmp_path: Path, suffix: str) -> None:
    lines = load_stored_array(path=_MD_CNN, name=_MD_CNN_ARRAY)
    path = tmp_path / f"lines{suffix}"
    if suffix == ".npy":
        np.save(file=path, arr=lines)
    else:
        np.savez(path, noise=np.zeros(shape=(3,)), lines=lines)

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot, layout=_LINE_LAYOUT)
        # A .npy file's single array is read at once; a .npz file's is chosen.
        await _load_lines(
            pilot=pilot, path=path, array=None if suffix == ".npy" else "lines"
        )
        assert isinstance(_screen(pilot), AcquisitionScreen)
        _set(pilot=pilot, axis_coil="3")
        await _press(pilot=pilot, button_id="read")
        screen = _screen(pilot)
        assert isinstance(screen, GridScreen)
        loaded = screen.session.loaded
        assert loaded is not None
        expected = select_radial_kspace(
            values=lines, layout=_LINE_LAYOUT, selection={"coil": 3}
        )
        np.testing.assert_array_equal(loaded.values, expected.values)

    _drive(steps=steps)


def test_an_array_in_another_layout_is_reported_on_the_array_screen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lines.npz"
    np.savez(path, noise=np.zeros(shape=(3,)))

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot, layout=_LINE_LAYOUT)
        await _load(pilot=pilot, path=path)
        assert isinstance(_screen(pilot), ArrayScreen)
        await _press(pilot=pilot, button_id="next")
        assert isinstance(_screen(pilot), ArrayScreen)
        assert "must be" in _status(pilot=pilot)

    _drive(steps=steps)


def test_a_mat_file_in_another_layout_is_reported_on_the_file_screen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "other.mat"
    path.write_bytes(data=b"not a MATLAB file")

    async def steps(pilot: Pilot) -> None:
        await _to_file(
            pilot=pilot, layout=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX
        )
        await _load(pilot=pilot, path=path)
        assert isinstance(_screen(pilot), FileScreen)
        assert _status(pilot=pilot).startswith("Error:")

    _drive(steps=steps)


def test_tracing_and_the_accurate_path(tmp_path: Path) -> None:
    path = _save_npy(path=tmp_path / "f.npy", values=np.exp(-(_GRID.r.T**2)))

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R)
        _screen(pilot).query_one("#trace", Checkbox).value = True
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        assert screen.session.result is not None
        assert len(screen.session.result.signals) == 4
        await _press(pilot=pilot, button_id="back")
        # The accurate path's inverse runs on grid samples, approximately.
        _screen(pilot).query_one("#trace", Checkbox).value = False
        _select(pilot=pilot, select_id="direction", value=Direction.INVERSE)
        _select(pilot=pilot, select_id="route", value=Route.ACCURATE)
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        result = screen.session.result
        assert result is not None
        assert any("approximation" in notice for notice in result.notices)

    _drive(steps=steps)


# ======================================================================================
# Invalid input
# ======================================================================================


def _assert_error_on(pilot: Pilot, screen_class: type[StepScreen]) -> None:
    """The current screen is still ``screen_class`` and shows an error."""
    assert isinstance(_screen(pilot), screen_class)
    assert _status(pilot=pilot).startswith("Error: "), _status(pilot=pilot)


def test_a_rank_that_does_not_match_the_choice_is_reported(tmp_path: Path) -> None:
    path = _save_npy(path=tmp_path / "single.npy", values=np.zeros(shape=(4, 3)))

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot, layout=SampleLayout.BATCH)
        await _load(pilot=pilot, path=path)
        _assert_error_on(pilot=pilot, screen_class=FileScreen)
        assert "expected (radial, angular, batch)" in _status(pilot=pilot)

    _drive(steps=steps)


def test_an_empty_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.npy"
    path.write_bytes(data=b"")

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        await _load(pilot=pilot, path=path)
        _assert_error_on(pilot=pilot, screen_class=FileScreen)
        assert "cannot be read" in _status(pilot=pilot)

    _drive(steps=steps)


def test_an_unsupported_suffix_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text(data="1")

    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        await _load(pilot=pilot, path=path)
        _assert_error_on(pilot=pilot, screen_class=FileScreen)

    _drive(steps=steps)


@pytest.mark.parametrize("R", ["-1", "0", "R"])
def test_an_invalid_R_is_reported(R: str) -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_file(pilot=pilot)
        await _load(pilot=pilot, path=_MAZE)
        _set(pilot=pilot, R=R)
        await _press(pilot=pilot, button_id="next")
        _assert_error_on(pilot=pilot, screen_class=GridScreen)

    _drive(steps=steps)


def test_a_real_image_chosen_for_the_ipft_is_reported() -> None:
    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=_MAZE, R=1.0)
        _select(pilot=pilot, select_id="direction", value=Direction.INVERSE)
        await _press(pilot=pilot, button_id="run")
        _assert_error_on(pilot=pilot, screen_class=DirectionScreen)
        assert "image" in _status(pilot=pilot)

    _drive(steps=steps)


def test_R_above_a_uniform_files_radius_is_reported(tmp_path: Path) -> None:
    path = _save_npy(path=tmp_path / "uniform.npy", values=np.ones(shape=(20, 5)))

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R, radius=_R / 2)
        await _press(pilot=pilot, button_id="run")
        _assert_error_on(pilot=pilot, screen_class=DirectionScreen)
        assert "grid.R" in _status(pilot=pilot)

    _drive(steps=steps)


# ======================================================================================
# Viewing and saving
# ======================================================================================


def test_a_batch_saves_one_pair_of_figures_per_element(tmp_path: Path) -> None:
    path = _batch_file(tmp_path=tmp_path, size=2)
    output = tmp_path / "figures"
    before = len(plt.get_fignums())

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R, layout=SampleLayout.BATCH)
        await _press(pilot=pilot, button_id="run")
        _set(pilot=pilot, directory=str(output))
        await _press(pilot=pilot, button_id="save")
        await _wait(pilot=pilot, done=lambda: "Saved" in _status(pilot=pilot))

    _drive(steps=steps)
    assert sorted(path.name for path in output.iterdir()) == sorted(
        f"batch_polar_frequency_angular_{part}_b{index:03d}.png"
        for index in range(2)
        for part in ("magnitude", "phase")
    )
    assert len(plt.get_fignums()) == before


def test_a_large_batch_shows_the_slowness_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_io, "SLOW_BATCH_SIZE", 1)
    path = _batch_file(tmp_path=tmp_path, size=2)

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R, layout=SampleLayout.BATCH)
        await _press(pilot=pilot, button_id="run")
        _set(pilot=pilot, directory=str(tmp_path / "figures"))
        await _press(pilot=pilot, button_id="save")
        await _wait(pilot=pilot, done=lambda: "Saved" in _status(pilot=pilot))
        assert "CLIPerformanceWarning" in _status(pilot=pilot)

    _drive(steps=steps)


def test_saving_asks_before_replacing_files(tmp_path: Path) -> None:
    path = _save_npy(path=tmp_path / "f.npy", values=np.exp(-(_GRID.r.T**2)))
    output = tmp_path / "figures"

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R)
        await _press(pilot=pilot, button_id="run")
        _set(pilot=pilot, directory=str(output))
        await _press(pilot=pilot, button_id="save")
        await _wait(pilot=pilot, done=lambda: "Saved" in _status(pilot=pilot))
        stamps = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
        await _press(pilot=pilot, button_id="save")
        assert isinstance(_screen(pilot), ConfirmScreen)
        await _press(pilot=pilot, button_id="no")
        await _wait(pilot=pilot, done=lambda: "Nothing saved" in _status(pilot=pilot))
        assert {p.name: p.stat().st_mtime_ns for p in output.iterdir()} == stamps
        await _press(pilot=pilot, button_id="save")
        await _press(pilot=pilot, button_id="yes")
        await _wait(pilot=pilot, done=lambda: "Saved 2 files" in _status(pilot=pilot))

    _drive(steps=steps)


def test_viewing_without_a_window_writes_one_figure_in_place(tmp_path: Path) -> None:
    path = _batch_file(tmp_path=tmp_path, size=3)
    before = len(plt.get_fignums())

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R, layout=SampleLayout.BATCH)
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        await _press(pilot=pilot, button_id="view")
        figure = screen.figure
        assert isinstance(figure, Figure)
        first = np.array(object=figure.axes[0].images[0].get_array())
        view_file = Path(_status(pilot=pilot).rsplit(" to ", 1)[-1])
        assert plt.imread(fname=view_file).ndim == 3
        # Another batch element, drawn into the same figure.
        _set(pilot=pilot, batch_index="2")
        await _press(pilot=pilot, button_id="view")
        assert screen.figure is figure
        second = np.array(object=figure.axes[0].images[0].get_array())
        assert not np.array_equal(a1=second, a2=first)
        assert len(plt.get_fignums()) == before
        _set(pilot=pilot, batch_index="3")
        await _press(pilot=pilot, button_id="view")
        _assert_error_on(pilot=pilot, screen_class=ResultScreen)

    _drive(steps=steps)
    assert len(plt.get_fignums()) == before


def test_viewing_in_a_window_reuses_one_figure_and_closes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pretend the backend opens windows; Agg would warn that it cannot show one.
    monkeypatch.setattr(_app, "interactive_backend", lambda: True)
    monkeypatch.setattr(Figure, "show", lambda self: None)
    path = _batch_file(tmp_path=tmp_path, size=2)
    before = len(plt.get_fignums())

    async def steps(pilot: Pilot) -> None:
        await _to_direction(pilot=pilot, path=path, R=_R, layout=SampleLayout.BATCH)
        await _press(pilot=pilot, button_id="run")
        screen = _screen(pilot)
        assert isinstance(screen, ResultScreen)
        await _press(pilot=pilot, button_id="view")
        figure = screen.figure
        assert len(plt.get_fignums()) == before + 1
        _set(pilot=pilot, batch_index="1")
        await _press(pilot=pilot, button_id="view")
        assert screen.figure is figure
        assert len(plt.get_fignums()) == before + 1
        await _press(pilot=pilot, button_id="back")
        assert len(plt.get_fignums()) == before

    _drive(steps=steps)
    assert len(plt.get_fignums()) == before
