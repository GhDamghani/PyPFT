"""A minimal terminal interface for converting a file of radial k-space lines.

``python -m pypft.acquisition [path]`` opens one screen: the file and the layout of
its array, an **Inspect** button that reads the array and lists its selectable axes
(``selectable_axes``), one field per axis taking an index (a single sample) or ``all``
(the whole axis, in the batch), and an output ``.npz`` path. A ``.npy`` file's single
array is read as it is; for a ``.npz`` or ``.mat`` file, Inspect also lists its arrays
in an array selector, and inspecting again reads the one chosen there. **Convert**
reads the selection with ``select_radial_kspace`` and writes it with
``RadialKSpace.save``; the status line shows the result or the error, and the app
keeps running either way. An existing output file is replaced only when **Overwrite**
is checked.

It runs in any terminal, with no graphical toolkit: Textual draws with text alone.
Everything it does is a few calls of the public API, so a script does the same
without it.
"""

import argparse
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from pypft.utils.validators import PathValidator, StrValidator

from . import (
    RadialLineLayout,
    holds_several_arrays,
    load_stored_array,
    select_radial_kspace,
    selectable_axes,
    stored_arrays,
)
from ._kspace import _KSPACE_SUFFIX

#: What an axis field accepts, besides an index, to keep the whole axis.
WHOLE_AXIS_WORD = "all"

#: The suffix appended to an input file's stem to name its default output file.
_OUTPUT_STEM_SUFFIX = "_kspace"

#: The prefix of every axis field's widget id; the axis name follows it.
_AXIS_ID_PREFIX = "axis-"

#: The widget id of the selector of a file's arrays.
_ARRAY_SELECT_ID = "array"

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


def array_label(name: str, shape: tuple[int, ...]) -> str:
    """Name one of a file's arrays for a selector: its name and shape.

    :param name: The array's name in its file.
    :type name: str
    :param shape: Its shape.
    :type shape: tuple[int, ...]
    :returns: The label.
    :rtype: str

    """
    return f"{name} {shape}"


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
        # The file and array last inspected, and the file whose arrays are listed.
        self._inspected: tuple[Path, str | None] | None = None
        self._listed: Path | None = None
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
        array_select: Select[str] = Select(options=[], id=_ARRAY_SELECT_ID)
        array_select.display = False
        yield array_select
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

    def _array_name(self, path: Path) -> str | None:
        """Choose the array to read: none for a single-array file, else the selected.

        A file holding several arrays fills the array selector with them when it is
        not yet listed there, and its first array is chosen.

        :param path: The input file.
        :type path: Path
        :returns: The array's name, or ``None`` for a file holding a single array.
        :rtype: str | None

        """
        select = self.query_one(selector=f"#{_ARRAY_SELECT_ID}", expect_type=Select)
        if not holds_several_arrays(path=path):
            select.display = False
            return None
        arrays = stored_arrays(path=path)
        if self._listed != path or select.value not in arrays:
            select.set_options(
                [
                    (array_label(name=name, shape=shape), name)
                    for name, shape in arrays.items()
                ]
            )
            select.value = next(iter(arrays))
            self._listed = path
        select.display = True
        return str(select.value)

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
        # A Text is shown as typed: textual 1.0 reads a str update as markup even
        # on a Static built with markup=False.
        self.query_one(selector="#status", expect_type=Static).update(Text(message))

    # ==================================================================================
    # Actions
    # ==================================================================================

    async def _inspect(self) -> None:
        """Read the file's array and list its selectable axes, one field per axis."""
        path = self._input_path()
        array_name = self._array_name(path=path)
        values = load_stored_array(path=path, name=array_name)
        self._axes = selectable_axes(values=values, layout=self._layout())
        self._inspected = (path, array_name)
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
        array = "" if array_name is None else f" {array_name!r}"
        self._report(f"{path.name}{array}: {self._axes}")

    def _convert(self) -> None:
        """Read the selection and save it, reporting the shape and radius."""
        path = self._input_path()
        if self._inspected is None or self._inspected[0] != path:
            raise ValueError("inspect the file before converting it")
        output_field = self.query_one(selector="#output", expect_type=Input)
        output = Path(output_field.value.strip())
        overwrite = self.query_one(selector="#overwrite", expect_type=Checkbox)
        if output.exists() and not overwrite.value:
            raise FileExistsError(f"{output} exists; check Overwrite to replace it")
        values = load_stored_array(path=path, name=self._inspected[1])
        kspace = select_radial_kspace(
            values=values,
            layout=self._layout(),
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
