"""A minimal terminal interface for converting a file of radial k-space lines.

``python -m pypft.acquisition [path]`` opens one screen: the file and its layout, an
**Inspect** button that lists the file's selectable axes (``selectable_axes``), one
field per axis taking an index (a single sample) or ``all`` (the whole axis, in the
batch), and an output ``.npz`` path. **Convert** reads the selection with
``read_radial_kspace`` and writes it with ``RadialKSpace.save``; the status line shows
the result or the error, and the app keeps running either way. An existing output
file is replaced only when **Overwrite** is checked.

It runs in any terminal, with no graphical toolkit: Textual draws with text alone.
Everything it does is two calls of the public API, so a script does the same without
it.
"""

import argparse
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from pypft.utils.validators import PathValidator, StrValidator

from . import (
    RadialLineLayout,
    layout_suffix,
    read_radial_kspace,
    selectable_axes,
)
from ._kspace import _KSPACE_SUFFIX

#: What an axis field accepts, besides an index, to keep the whole axis.
WHOLE_AXIS_WORD = "all"

#: The suffix appended to an input file's stem to name its default output file.
_OUTPUT_STEM_SUFFIX = "_kspace"

#: The prefix of every axis field's widget id; the axis name follows it.
_AXIS_ID_PREFIX = "axis-"

#: The errors a conversion reports on screen instead of ending the app: every
#: validation and file error the public API raises.
_REPORTED_ERRORS = (OSError, TypeError, ValueError)


def parse_axis_choice(text: str) -> int | None:
    """Read one axis field: an index, or ``all`` for the whole axis.

    :param text: The field's text; surrounding spaces and case are ignored.
    :type text: str
    :returns: The index, or ``None`` for the whole axis.
    :rtype: int | None
    :raises TypeError: If ``text`` is not a ``str``.
    :raises ValueError: If the text is neither an integer nor ``all``.

    """
    StrValidator.type_is_str(value=text)
    cleaned = text.strip().lower()
    if cleaned == WHOLE_AXIS_WORD:
        return None
    try:
        return int(cleaned)
    except ValueError:
        raise ValueError(
            f"an axis takes an index or {WHOLE_AXIS_WORD!r}, got {text!r}"
        ) from None


def default_output_path(path: Path) -> Path:
    """Name the default output file of an input file: ``<stem>_kspace.npz`` beside it.

    :param path: The input file.
    :type path: Path
    :returns: The output path.
    :rtype: Path
    :raises TypeError: If ``path`` is not a ``pathlib.Path``.

    """
    PathValidator.type_is_Path(value=path)
    return path.with_name(f"{path.stem}{_OUTPUT_STEM_SUFFIX}{_KSPACE_SUFFIX}")


class AcquisitionApp(App[None]):
    """The single-screen converter from a file of radial lines to a ``.npz`` file.

    :param path: The file to open with, if any.
    :type path: Path | None
    :raises TypeError: If ``path`` is neither a ``pathlib.Path`` nor ``None``.

    """

    TITLE = "PyPFT acquisition converter"

    CSS = """
    Vertical { height: auto; }
    Horizontal { height: auto; }
    Input { width: 1fr; }
    #status { margin: 1 0; }
    """

    def __init__(self, path: Path | None = None) -> None:
        """Remember the file to open with."""
        if path is not None:
            PathValidator.type_is_Path(value=path)
        super().__init__()
        self._initial_path = "" if path is None else str(path)
        self._axes: dict[str, int] = {}
        self._status = ""

    @property
    def status(self) -> str:
        """The message the status line shows: the last result or error.

        :returns: The message.
        :rtype: str

        """
        return self._status

    def compose(self) -> ComposeResult:
        """Lay out the screen.

        :returns: The widgets, top to bottom.
        :rtype: ComposeResult

        """
        # Label and Static take their text positionally: its parameter is named
        # differently across the supported textual releases.
        yield Label("File of radial k-space lines")
        yield Input(value=self._initial_path, placeholder="path/to/file", id="path")
        yield Select(
            options=[(member.name, member) for member in RadialLineLayout],
            value=next(iter(RadialLineLayout)),
            allow_blank=False,
            id="layout",
        )
        yield Button(label="Inspect", id="inspect")
        yield Vertical(id="axes")
        yield Label("Output file (.npz)")
        yield Input(placeholder="path/to/output.npz", id="output")
        yield Checkbox(label="Overwrite an existing output file", id="overwrite")
        with Horizontal():
            yield Button(label="Convert", id="convert", variant="primary")
            yield Button(label="Quit", id="quit")
        # Paths and error messages are shown as typed, never read as markup.
        yield Static(id="status", markup=False)

    # ==================================================================================
    # Reading the screen
    # ==================================================================================

    def _input_path(self) -> Path:
        """Read the file field.

        :returns: The input file's path.
        :rtype: Path

        """
        return Path(self.query_one(selector="#path", expect_type=Input).value.strip())

    def _layout(self) -> RadialLineLayout:
        """Read the layout selector.

        :returns: The selected layout.
        :rtype: RadialLineLayout

        """
        return RadialLineLayout(
            self.query_one(selector="#layout", expect_type=Select).value
        )

    def _selection(self) -> dict[str, int | None]:
        """Read every axis field.

        :returns: The selection, one entry per inspected axis.
        :rtype: dict[str, int | None]
        :raises ValueError: If a field holds neither an index nor ``all``.

        """
        return {
            name: parse_axis_choice(
                text=self.query_one(
                    selector=f"#{_AXIS_ID_PREFIX}{name}", expect_type=Input
                ).value
            )
            for name in self._axes
        }

    def _report(self, message: str) -> None:
        """Show one message on the status line.

        :param message: The message.
        :type message: str

        """
        self._status = message
        self.query_one(selector="#status", expect_type=Static).update(message)

    # ==================================================================================
    # Actions
    # ==================================================================================

    async def _inspect(self) -> None:
        """List the file's selectable axes, one field per axis."""
        path = self._input_path()
        self._axes = selectable_axes(path=path, layout=self._layout())
        container = self.query_one(selector="#axes", expect_type=Vertical)
        await container.remove_children()
        for name, size in self._axes.items():
            await container.mount(
                Label(f"{name}: 0 .. {size - 1}, or {WHOLE_AXIS_WORD}"),
                Input(value="0", id=f"{_AXIS_ID_PREFIX}{name}"),
            )
        output = self.query_one(selector="#output", expect_type=Input)
        if not output.value.strip():
            output.value = str(default_output_path(path=path))
        self._report(f"{path.name}: {self._axes}")

    def _convert(self) -> None:
        """Read the selection and save it, reporting the shape and radius."""
        path = self._input_path()
        layout = self._layout()
        PathValidator.value_has_correct_suffix(
            value=path, suffix=layout_suffix(layout=layout)
        )
        if not self._axes:
            raise ValueError("inspect the file before converting it")
        output_field = self.query_one(selector="#output", expect_type=Input)
        output = Path(output_field.value.strip())
        overwrite = self.query_one(selector="#overwrite", expect_type=Checkbox)
        if output.exists() and not overwrite.value:
            raise FileExistsError(f"{output} exists; check Overwrite to replace it")
        kspace = read_radial_kspace(
            path=path,
            layout=layout,
            selection=self._selection(),
        )
        kspace.save(path=output)
        self._report(
            f"Saved {kspace.values.shape} (radius {kspace.radius:.6g} rad/pixel) "
            f"to {output}"
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Run the pressed button's action, reporting any error on screen.

        :param event: The button press.
        :type event: Button.Pressed

        """
        if event.button.id == "quit":
            self.exit()
            return
        try:
            if event.button.id == "inspect":
                await self._inspect()
            elif event.button.id == "convert":
                self._convert()
        except _REPORTED_ERRORS as error:
            self._report(f"Error: {error}")


def main(argv: list[str] | None = None) -> int:
    """Run the converter from the command line.

    :param argv: The command-line arguments; ``sys.argv[1:]`` when omitted.
    :type argv: list[str] | None
    :returns: The exit code, ``0``.
    :rtype: int

    """
    parser = argparse.ArgumentParser(
        prog="python -m pypft.acquisition",
        description="Convert radial k-space lines into PyPFT's uniform polar k-space.",
    )
    parser.add_argument(
        "path", nargs="?", type=Path, default=None, help="The file to open with."
    )
    arguments = parser.parse_args(args=argv)
    AcquisitionApp(path=arguments.path).run()
    return 0
