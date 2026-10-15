"""Tests for ``pypft.acquisition.app``, the acquisition converter's terminal interface.

The app is driven headlessly through Textual's ``App.run_test``, inside
``asyncio.run`` so no pytest plugin is needed. A button is pressed through
``Button.press`` (one test also clicks with the mouse): a click is ignored while the
button still shows its previous click, which would make repeated presses depend on
timing. The status line is read through ``AcquisitionApp.status``, every other
screen state from the widgets themselves.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest
from textual.pilot import Pilot
from textual.widgets import Button, Checkbox, Input

from pypft.acquisition import RadialKSpace, RadialLineLayout, read_radial_kspace
from pypft.acquisition.app import (
    AcquisitionApp,
    default_output_path,
    main,
    parse_axis_choice,
)

#: The committed excerpt of the MD-CNN dataset: one slice, one phase, four coils.
_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)

#: A terminal tall enough to show every widget at once.
_SIZE = (100, 50)

#: How long a button press may take to report, and how often that is checked, in
#: seconds; a conversion reads the fixture, well under a second.
_REPORT_TIMEOUT, _REPORT_POLL = 30.0, 0.05


def _drive(app: AcquisitionApp, steps: Callable[[Pilot], Awaitable[None]]) -> None:
    """Run ``app`` headlessly and drive it with ``steps``.

    :param app: The app.
    :type app: AcquisitionApp
    :param steps: What to do with the running app's pilot.
    :type steps: Callable[[Pilot], Awaitable[None]]

    """

    async def run() -> None:
        async with app.run_test(size=_SIZE) as pilot:
            await steps(pilot)

    asyncio.run(run())


async def _press(pilot: Pilot, button_id: str) -> None:
    """Press a button and wait until its handler has reported on the status line.

    Every button but **Quit** ends by reporting, and an async handler may still be
    mounting widgets when the pilot first finds the app idle, so the status line,
    not idleness, marks the end.

    :param pilot: The running app's pilot.
    :type pilot: Pilot
    :param button_id: The button's id, without ``#``.
    :type button_id: str
    :raises AssertionError: If nothing is reported within ``_REPORT_TIMEOUT``.

    """
    app = pilot.app
    assert isinstance(app, AcquisitionApp)
    before = app.status
    app.query_one(f"#{button_id}", Button).press()
    for _ in range(int(_REPORT_TIMEOUT / _REPORT_POLL)):
        await pilot.pause(delay=_REPORT_POLL)
        if app.status != before:
            return
    raise AssertionError(f"pressing {button_id!r} reported nothing")


def _set(app: AcquisitionApp, widget_id: str, value: str) -> None:
    """Type a value into an input field.

    :param app: The app.
    :type app: AcquisitionApp
    :param widget_id: The field's id, without ``#``.
    :type widget_id: str
    :param value: The text.
    :type value: str

    """
    app.query_one(f"#{widget_id}", Input).value = value


# ======================================================================================
# The pure helpers
# ======================================================================================


@pytest.mark.parametrize(
    ("text", "expected"), [("0", 0), (" 3 ", 3), ("all", None), (" ALL ", None)]
)
def test_parse_axis_choice(text: str, expected: int | None) -> None:
    assert parse_axis_choice(text=text) == expected


@pytest.mark.parametrize("text", ["", "1.5", "every"])
def test_parse_axis_choice_rejects_other_text(text: str) -> None:
    with pytest.raises(ValueError, match="index or 'all'"):
        parse_axis_choice(text=text)


def test_arguments_are_type_validated() -> None:
    with pytest.raises(TypeError):
        parse_axis_choice(text=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        default_output_path(path="P01.mat")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AcquisitionApp(path="P01.mat")  # type: ignore[arg-type]


def test_default_output_path_sits_beside_the_input() -> None:
    assert default_output_path(path=Path("data") / "P01.mat") == (
        Path("data") / "P01_kspace.npz"
    )


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv=["--help"])
    assert exit_info.value.code == 0
    assert "python -m pypft.acquisition" in capsys.readouterr().out


# ======================================================================================
# The screen
# ======================================================================================


def test_inspect_lists_every_axis_and_proposes_an_output() -> None:
    app = AcquisitionApp(path=_FIXTURE)

    async def steps(pilot: Pilot) -> None:
        await pilot.click("#inspect")
        for _ in range(int(_REPORT_TIMEOUT / _REPORT_POLL)):
            if app.status:
                break
            await pilot.pause(delay=_REPORT_POLL)
        for name in ("slice", "phase", "coil"):
            assert app.query_one(f"#axis-{name}", Input).value == "0"
        assert app.query_one("#output", Input).value == str(
            default_output_path(path=_FIXTURE)
        )
        assert "'coil': 4" in app.status

    _drive(app=app, steps=steps)


@pytest.mark.parametrize(
    ("coil", "selection"), [("2", {"coil": 2}), ("all", {"coil": None})]
)
def test_convert_saves_a_single_sample_or_a_batch(
    tmp_path: Path, coil: str, selection: dict[str, int | None]
) -> None:
    app = AcquisitionApp(path=_FIXTURE)
    output = tmp_path / "kspace.npz"

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="inspect")
        _set(app=app, widget_id="axis-coil", value=coil)
        _set(app=app, widget_id="output", value=str(output))
        await _press(pilot=pilot, button_id="convert")
        assert app.status.startswith("Saved")

    _drive(app=app, steps=steps)
    expected = read_radial_kspace(
        path=_FIXTURE,
        layout=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX,
        selection=selection,
    )
    saved = RadialKSpace.load(path=output)
    np.testing.assert_array_equal(saved.values, expected.values)
    assert saved.radius == expected.radius


def test_an_existing_output_is_replaced_only_when_asked(tmp_path: Path) -> None:
    app = AcquisitionApp(path=_FIXTURE)
    output = tmp_path / "kspace.npz"
    output.write_bytes(data=b"keep")

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="inspect")
        _set(app=app, widget_id="output", value=str(output))
        await _press(pilot=pilot, button_id="convert")
        assert "Overwrite" in app.status
        assert output.read_bytes() == b"keep"
        app.query_one("#overwrite", Checkbox).value = True
        await _press(pilot=pilot, button_id="convert")
        assert app.status.startswith("Saved")

    _drive(app=app, steps=steps)
    assert RadialKSpace.load(path=output).values.shape == (208, 392)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("axis-coil", "every", "index or 'all'"),
        ("axis-coil", "9", "less than 4"),
        ("path", "missing.mat", "missing.mat"),
        ("path", "notes.txt", "suffix"),
    ],
)
def test_errors_are_shown_and_the_app_keeps_running(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    app = AcquisitionApp(path=_FIXTURE)

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="inspect")
        _set(app=app, widget_id="output", value=str(tmp_path / "kspace.npz"))
        _set(app=app, widget_id=field, value=value)
        await _press(pilot=pilot, button_id="convert")
        assert app.status.startswith("Error")
        assert message in app.status
        assert app.is_running

    _drive(app=app, steps=steps)
    assert not (tmp_path / "kspace.npz").exists()


def test_an_unreadable_file_is_reported_and_the_app_keeps_running(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.mat"
    broken.write_bytes(data=b"")
    app = AcquisitionApp(path=broken)

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="inspect")
        assert "not a readable MATLAB 5 file" in app.status
        assert app.is_running

    _drive(app=app, steps=steps)


def test_the_status_line_shows_brackets_as_typed(tmp_path: Path) -> None:
    app = AcquisitionApp(path=_FIXTURE)

    async def steps(pilot: Pilot) -> None:
        await _press(pilot=pilot, button_id="inspect")
        _set(app=app, widget_id="output", value=str(tmp_path / "kspace.npz"))
        # A closing tag with no opening one would fail to render as markup.
        _set(app=app, widget_id="axis-coil", value="[/v2]")
        await _press(pilot=pilot, button_id="convert")
        await pilot.pause()
        assert "[/v2]" in app.status
        assert app.is_running

    _drive(app=app, steps=steps)


def test_convert_before_inspect_is_an_error(tmp_path: Path) -> None:
    app = AcquisitionApp(path=_FIXTURE)

    async def steps(pilot: Pilot) -> None:
        _set(app=app, widget_id="output", value=str(tmp_path / "kspace.npz"))
        await _press(pilot=pilot, button_id="convert")
        assert "inspect" in app.status

    _drive(app=app, steps=steps)


def test_quit_ends_the_app() -> None:
    app = AcquisitionApp()

    async def steps(pilot: Pilot) -> None:
        app.query_one("#quit", Button).press()
        await pilot.pause()
        assert not app.is_running

    _drive(app=app, steps=steps)
    assert app.return_code == 0
