"""The terminal interface's screens: ``PyPFTApp``, one screen per step of the flow.

The flow is **Start** (transform a file, or open the tutorials) -> **Layout** (a
single sample, a batch, or radial k-space lines in one of
``pypft.acquisition.RadialLineLayout``'s layouts) -> **File** (typed, or picked
from a filtered directory tree) -> **Acquisition** (radial lines only: one index or
``all`` per selectable axis, as ``pypft.acquisition.read_radial_kspace`` takes) ->
**Grid**
(where the samples sit, ``R``, the limit kind; the adequacy checks run here) ->
**Direction** (PFT or IPFT, the exact or the accurate path, optional tracing) ->
**Result** (view or save the figures). Every screen but the first has **Back**, and
every screen has **Quit** and a status line; an error is shown there and the app keeps
running.

This module holds the user interface only: loading and saving go through
``pypft.cli._io``, transforming through ``pypft.cli._pipeline``.
"""

import importlib.util
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Input,
    Label,
    Select,
    Static,
)

from pypft.acquisition import RadialLineLayout, selectable_axes
from pypft.acquisition.app import WHOLE_AXIS_WORD, parse_axis_choice
from pypft.domains import BaseSignal, PolarDomain
from pypft.grid import LimitKind
from pypft.transform import Direction

from ._io import (
    SUPPORTED_SUFFIXES,
    CLIPerformanceWarning,
    LoadedInput,
    batch_size,
    draw_signal,
    figure_paths,
    interactive_backend,
    load_acquisition,
    load_array,
    record_warnings,
    sample_of,
    save_signal_figures,
    view_figure,
    warn_if_slow,
)
from ._pipeline import (
    GridSettings,
    InputGrid,
    Route,
    TransformResult,
    check_grid,
    parse_optional_positive_float,
    parse_positive_float,
    parse_positive_int,
    run_transform,
    suggested_grid,
)

# ======================================================================================
# Constants
# ======================================================================================

#: An enum whose members a selector offers.
E = TypeVar("E", bound=Enum)

#: The online tutorials: the rendered notebooks, which are not part of the package.
TUTORIALS_URL = "https://pypft.readthedocs.io/en/latest/tutorials.html"

#: The tutorial notebooks of a clone of the repository (absent from an installed
#: package): ``src/pypft/cli/_app.py`` sits three directories below the clone's root.
_NOTEBOOKS_DIRECTORY = Path(__file__).resolve().parents[3] / "notebooks"

#: The module that opens the notebooks, when the development tools are installed.
_JUPYTERLAB_MODULE = "jupyterlab"

#: The errors a screen reports on its status line instead of ending the app: every
#: validation, file and unsupported-case error the public API raises.
_REPORTED_ERRORS = (OSError, TypeError, ValueError, NotImplementedError)

#: How often an open figure window gets to process its events, in seconds: the app,
#: not matplotlib, owns the event loop.
_FIGURE_EVENTS_INTERVAL = 0.1

#: The file a viewed figure is written to when no window can open.
_VIEW_FILE_NAME = "view.png"

#: The end of a status line shown while a long step runs; the step's own report
#: replaces it when the step ends.
_BUSY_SUFFIX = "..."

#: How a recorded slowness warning starts (``pypft.cli._io.record_warnings``).
_SLOWNESS_PREFIX = f"{CLIPerformanceWarning.__name__}:"

#: The default output directory's name: the input file's stem, then this.
_FIGURES_DIRECTORY_SUFFIX = "_figures"

#: The prefix of every acquisition axis field's widget id; the axis name follows.
_AXIS_ID_PREFIX = "axis-"

#: The PFT and IPFT, as the direction selector names them.
_DIRECTION_LABELS = (
    ("PFT: spatial input to frequency output", Direction.FORWARD),
    ("IPFT: frequency input to spatial output", Direction.INVERSE),
)


class SampleLayout(Enum):
    """Whether a file holds a single sample or a batch of samples."""

    SINGLE = "a single sample (radial, angular)"
    BATCH = "a batch of samples (radial, angular, batch)"


#: What the layout selector offers: a polar array file's layout, or one of the
#: layouts of radial k-space lines ``pypft.acquisition`` reads, whose own selection
#: then makes a single sample or a batch.
InputLayout = SampleLayout | RadialLineLayout


def layout_options() -> list[tuple[str, InputLayout]]:
    """List the layout selector's options: every polar layout, then every line layout.

    :returns: Each option's label and value.
    :rtype: list[tuple[str, InputLayout]]

    """
    options: list[tuple[str, InputLayout]] = [
        (layout.value, layout) for layout in SampleLayout
    ]
    options.extend(
        (f"radial k-space lines {member.name}", member) for member in RadialLineLayout
    )
    return options


def layout_description(layout: InputLayout) -> str:
    """Describe what a file of a layout holds.

    :param layout: The layout.
    :type layout: InputLayout
    :returns: The description.
    :rtype: str

    """
    if isinstance(layout, SampleLayout):
        return layout.value
    return f"radial k-space lines in the {layout.name} layout"


# ======================================================================================
# Shared state and helpers
# ======================================================================================


@dataclass
class Session:
    """Everything chosen so far, shared by every screen.

    :param layout: A single sample, a batch, or a layout of radial lines.
    :type layout: InputLayout
    :param path: The input file.
    :type path: Path | None
    :param loaded: The array read from it.
    :type loaded: pypft.cli._io.LoadedInput | None
    :param settings: The grid's description.
    :type settings: pypft.cli._pipeline.GridSettings | None
    :param grid_notices: The adequacy checks' warnings, as text.
    :type grid_notices: tuple[str, ...]
    :param result: The finished transform.
    :type result: pypft.cli._pipeline.TransformResult | None

    """

    layout: InputLayout = SampleLayout.SINGLE
    path: Path | None = None
    loaded: LoadedInput | None = None
    settings: GridSettings | None = None
    grid_notices: tuple[str, ...] = ()
    result: TransformResult | None = None


def notebooks_directory() -> Path | None:
    """Find tutorial notebooks that can be opened locally.

    :returns: The clone's ``notebooks`` directory, when it exists and JupyterLab is
        installed; else ``None``.
    :rtype: Path | None

    """
    if not _NOTEBOOKS_DIRECTORY.is_dir():
        return None
    if importlib.util.find_spec(name=_JUPYTERLAB_MODULE) is None:
        return None
    return _NOTEBOOKS_DIRECTORY


def _lines(*parts: Iterable[str]) -> str:
    """Join groups of text lines into one block, one line each."""
    return "\n".join(line for part in parts for line in part)


def listed_paths(paths: Iterable[Path]) -> list[Path]:
    """Keep the visible folders and the files with a supported suffix.

    :param paths: The entries of one folder.
    :type paths: Iterable[Path]
    :returns: The entries the file picker lists.
    :rtype: list[Path]

    """
    return [
        path
        for path in paths
        if not path.name.startswith(".")
        and (path.is_dir() or path.suffix.lower() in SUPPORTED_SUFFIXES)
    ]


class PolarFileTree(DirectoryTree):
    """A directory tree listing only folders and the files PyPFT can read."""

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        """Keep the entries ``listed_paths`` keeps.

        :param paths: The entries of one folder.
        :type paths: Iterable[Path]
        :returns: The entries to list.
        :rtype: Iterable[Path]

        """
        return listed_paths(paths=paths)


# ======================================================================================
# Screen scaffolding
# ======================================================================================


class StepScreen(Screen[None]):
    """One step of the flow: its own widgets, then Back, Quit and a status line.

    A subclass yields its widgets from ``compose_step`` and handles its own buttons
    in ``act``; an error raised there is shown on the status line.
    """

    #: The heading shown at the top of the screen.
    HEADING = ""

    #: Whether the screen has a Back button: every screen but the first.
    HAS_BACK = True

    def __init__(self) -> None:
        """Start with an empty status line."""
        super().__init__()
        self._status_text = ""

    @property
    def status(self) -> str:
        """The message the status line shows: the last result or error.

        :returns: The message.
        :rtype: str

        """
        return self._status_text

    @property
    def session(self) -> Session:
        """The choices made so far, shared by every screen.

        :returns: The app's session.
        :rtype: Session

        """
        return cast(PyPFTApp, self.app).session

    def compose(self) -> ComposeResult:
        """Lay out the heading, the step's widgets, the navigation and the status.

        :returns: The widgets, top to bottom.
        :rtype: ComposeResult

        """
        # Label and Static take their text positionally: its parameter is named
        # differently across the supported textual releases.
        yield Label(self.HEADING, id="heading")
        with VerticalScroll(id="body"):
            yield from self.compose_step()
        with Horizontal(id="navigation"):
            if self.HAS_BACK:
                yield Button(label="Back", id="back")
            yield Button(label="Quit", id="quit")
        # Paths and error messages are shown as typed, never read as markup.
        yield Static(id="status", markup=False)

    def compose_step(self) -> ComposeResult:
        """Yield the step's own widgets.

        :returns: The widgets, top to bottom.
        :rtype: ComposeResult

        """
        yield from ()

    def report(self, message: str) -> None:
        """Show one message on the status line.

        :param message: The message.
        :type message: str

        """
        self._status_text = message
        self.query_one(selector="#status", expect_type=Static).update(message)

    def act(self, button_id: str) -> None:
        """Run the action of one of the step's own buttons.

        :param button_id: The pressed button's id.
        :type button_id: str

        """

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Navigate, or run the step's action, reporting any error on screen.

        :param event: The button press.
        :type event: Button.Pressed

        """
        button_id = event.button.id or ""
        if button_id == "back":
            self.app.pop_screen()
        elif button_id == "quit":
            self.app.exit(result=0)
        else:
            try:
                self.act(button_id=button_id)
            except _REPORTED_ERRORS as error:
                self.report(message=f"Error: {error}")

    def _field(self, field_id: str) -> str:
        """Read a text field.

        :param field_id: The field's id, without ``#``.
        :type field_id: str
        :returns: The field's text.
        :rtype: str

        """
        return self.query_one(selector=f"#{field_id}", expect_type=Input).value

    def _choice(self, select_id: str, enum_class: type[E]) -> E:
        """Read a selector whose options are members of one enum.

        :param select_id: The selector's id, without ``#``.
        :type select_id: str
        :param enum_class: The enum its options belong to.
        :type enum_class: type[E]
        :returns: The selected member.
        :rtype: E
        :raises ValueError: If nothing is selected.

        """
        value = self.query_one(selector=f"#{select_id}", expect_type=Select).value
        if not isinstance(value, enum_class):
            raise ValueError(f"choose a {select_id.replace('-', ' ')}")
        return value


def _enum_select(
    options: Iterable[tuple[str, E]], value: E, select_id: str
) -> Select[E]:
    """Build a selector over enum members, one always selected.

    :param options: Each option's label and member.
    :type options: Iterable[tuple[str, E]]
    :param value: The member selected at first.
    :type value: E
    :param select_id: The selector's id.
    :type select_id: str
    :returns: The selector.
    :rtype: Select[E]

    """
    return Select(options=list(options), value=value, allow_blank=False, id=select_id)


# ======================================================================================
# Start and tutorials
# ======================================================================================


class StartScreen(StepScreen):
    """The first screen: transform a file, or open the tutorials."""

    HEADING = "PyPFT: the polar Fourier transform"
    HAS_BACK = False

    def compose_step(self) -> ComposeResult:
        """Yield the two choices.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        with Vertical(id="choices"):
            yield Button(label="Transform a file", id="transform", variant="primary")
            yield Button(label="Tutorials", id="tutorials")

    def act(self, button_id: str) -> None:
        """Open the chosen step.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "transform":
            self.app.push_screen(screen=SampleScreen())
        elif button_id == "tutorials":
            self.app.push_screen(screen=TutorialsScreen())


class TutorialsScreen(StepScreen):
    """Open the tutorials: online, or as notebooks from a clone of the repository."""

    HEADING = "Tutorials"

    def compose_step(self) -> ComposeResult:
        """Yield the tutorial links available here.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        yield Static(f"The tutorials are online at {TUTORIALS_URL}", markup=False)
        yield Button(label="Open the online tutorials", id="open-url")
        directory = notebooks_directory()
        if directory is not None:
            yield Static(f"Notebooks found in {directory}", markup=False)
            yield Button(label="Open the notebooks in JupyterLab", id="jupyter")

    def act(self, button_id: str) -> None:
        """Open the tutorials in a browser, or the notebooks in JupyterLab.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "open-url":
            webbrowser.open(url=TUTORIALS_URL)
            self.report(message=f"Opened {TUTORIALS_URL}")
        elif button_id == "jupyter":
            directory = notebooks_directory()
            if directory is None:
                raise ValueError("no notebooks to open, or JupyterLab is not installed")
            # JupyterLab runs on its own, independent of this app; its log output
            # is discarded, since it would be written over this app's screen.
            subprocess.Popen(
                args=[sys.executable, "-m", _JUPYTERLAB_MODULE, str(directory)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.report(message=f"Started JupyterLab on {directory}")


# ======================================================================================
# Sample or batch, and the file
# ======================================================================================


class SampleScreen(StepScreen):
    """Choose whether the file holds a single sample, a batch, or radial lines."""

    HEADING = "A single sample, a batch, or radial k-space lines?"

    def compose_step(self) -> ComposeResult:
        """Yield the layout selector.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        yield Label("The file holds")
        yield _enum_select(
            options=layout_options(), value=self.session.layout, select_id="layout"
        )
        yield Static(
            "Radial k-space lines are stored over several axes besides the readout "
            "and the line (slices, phases, coils, ...); after the file, one index of "
            "each makes a single sample, and keeping axes whole makes a batch.",
            id="acquisition-note",
            markup=False,
        )
        yield Button(label="Next", id="next", variant="primary")

    def act(self, button_id: str) -> None:
        """Record the layout and go on to the file.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "next":
            layout = self.query_one(selector="#layout", expect_type=Select).value
            if not isinstance(layout, (SampleLayout, RadialLineLayout)):
                raise ValueError("choose a layout")
            self.session.layout = layout
            self.app.push_screen(screen=FileScreen())


class FilePickerScreen(ModalScreen[Path | None]):
    """Pick a supported file from a directory tree.

    :param directory: The directory the tree starts at.
    :type directory: Path

    """

    def __init__(self, directory: Path) -> None:
        """Remember the starting directory."""
        super().__init__()
        self._directory = directory

    def compose(self) -> ComposeResult:
        """Lay out the tree and a Cancel button.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        with Vertical(id="dialog"):
            yield Label(f"Choose a file ({', '.join(SUPPORTED_SUFFIXES)})")
            yield PolarFileTree(path=self._directory, id="tree")
            yield Button(label="Cancel", id="cancel")

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """Return the selected file.

        :param event: The selection.
        :type event: DirectoryTree.FileSelected

        """
        self.dismiss(result=event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return without a file.

        :param event: The button press.
        :type event: Button.Pressed

        """
        if event.button.id == "cancel":
            self.dismiss(result=None)


class FileScreen(StepScreen):
    """Choose the input file, and read it."""

    HEADING = "Input file"

    def compose_step(self) -> ComposeResult:
        """Yield the file field and its buttons.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        yield Label(f"A file holding {layout_description(self.session.layout)}")
        initial = "" if self.session.path is None else str(self.session.path)
        yield Input(value=initial, placeholder="path/to/file", id="path")
        with Horizontal():
            yield Button(label="Browse", id="browse")
            yield Button(label="Load", id="load", variant="primary")

    def _picked(self, path: Path | None) -> None:
        """Put a picked file into the file field.

        :param path: The picked file, or ``None`` when cancelled.
        :type path: Path | None

        """
        if path is not None:
            self.query_one(selector="#path", expect_type=Input).value = str(path)
            self.report(message=f"Picked {path.name}")

    def act(self, button_id: str) -> None:
        """Open the picker, or read the file and go on to the grid.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "browse":
            typed = Path(self._field(field_id="path").strip())
            start = typed.parent if typed.parent.is_dir() else Path.cwd()
            self.app.push_screen(
                screen=FilePickerScreen(directory=start.resolve()),
                callback=self._picked,
            )
        elif button_id == "load":
            path = Path(self._field(field_id="path").strip())
            layout = self.session.layout
            if isinstance(layout, RadialLineLayout):
                # The header is read here, so a wrong file is reported on this screen.
                axes = selectable_axes(path=path, layout=layout)
                self.report(message=f"{path.name}: {axes}")
                self.app.push_screen(
                    screen=AcquisitionScreen(path=path, layout=layout, axes=axes)
                )
                return
            loaded = load_array(
                path=path, batched=self.session.layout is SampleLayout.BATCH
            )
            self.session.path, self.session.loaded = path, loaded
            self.report(message=f"Read {loaded.values.shape} from {path.name}")
            self.app.push_screen(screen=GridScreen())


class AcquisitionScreen(StepScreen):
    """Select the part of a file of radial lines to read, and read it.

    One field per selectable axis takes an index, or ``all`` to keep the whole axis
    in the batch; every field starts at index ``0``, a single sample.

    :param path: The file of radial lines.
    :type path: Path
    :param layout: Its layout, chosen on the layout screen.
    :type layout: pypft.acquisition.RadialLineLayout
    :param axes: Its selectable axes and their sizes.
    :type axes: dict[str, int]

    """

    HEADING = "Acquisition: which samples to read"

    def __init__(
        self, path: Path, layout: RadialLineLayout, axes: dict[str, int]
    ) -> None:
        """Remember the file, its layout and its axes."""
        super().__init__()
        self._path = path
        self._line_layout = layout
        self._axes = axes

    def compose_step(self) -> ComposeResult:
        """Yield one field per selectable axis, and the Read button.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        yield Static(
            f"{self._path.name} holds radial k-space lines ({self._line_layout.name}); "
            "they are converted into uniform polar k-space. One index of every axis "
            f"reads a single sample; {WHOLE_AXIS_WORD!r} keeps an axis whole, in a "
            "batch.",
            markup=False,
        )
        for name, size in self._axes.items():
            yield Label(f"{name}: 0 .. {size - 1}, or {WHOLE_AXIS_WORD}")
            yield Input(value="0", id=f"{_AXIS_ID_PREFIX}{name}")
        yield Button(label="Read", id="read", variant="primary")

    def _selection(self) -> dict[str, int | None]:
        """Read every axis field.

        :returns: The selection, one entry per selectable axis.
        :rtype: dict[str, int | None]
        :raises ValueError: If a field holds neither an index nor ``all``.

        """
        return {
            name: parse_axis_choice(
                text=self._field(field_id=f"{_AXIS_ID_PREFIX}{name}")
            )
            for name in self._axes
        }

    def act(self, button_id: str) -> None:
        """Read the selection and go on to the grid.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "read":
            loaded = load_acquisition(
                path=self._path,
                layout=self._line_layout,
                selection=self._selection(),
            )
            self.session.path, self.session.loaded = self._path, loaded
            self.report(
                message=f"Read {loaded.values.shape} from {self._path.name}, radius "
                f"{loaded.radius:.6g} rad/pixel"
            )
            self.app.push_screen(screen=GridScreen())


# ======================================================================================
# The grid, and the direction
# ======================================================================================


class GridScreen(StepScreen):
    """Describe the grid the file's samples sit on, and check it."""

    HEADING = "The grid"

    #: The fields only uniform polar input uses.
    _UNIFORM_FIELDS = ("n-radial", "n-angular", "radius")

    def _loaded(self) -> LoadedInput:
        """Return the array read on the file screen, or raise if none was read."""
        loaded = self.session.loaded
        if loaded is None:
            raise ValueError("no file has been read")
        return loaded

    def compose_step(self) -> ComposeResult:
        """Yield the grid's fields, filled from the file where it tells.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        loaded = self._loaded()
        n_radial, n_angular = loaded.values.shape[:2]
        # A .npz or acquisition file is uniform polar k-space carrying its radius.
        uniform = loaded.radius is not None
        yield Static(f"The array's samples: {n_radial} radial x {n_angular} angular")
        R = ""
        if loaded.radius is not None:
            suggestion = suggested_grid(shape=loaded.values.shape, radius=loaded.radius)
            n_radial, n_angular = suggestion.n_radial, suggestion.n_angular
            R = repr(suggestion.R)
            yield Static(
                f"Suggested for this k-space: R = {suggestion.R:.6g} pixels (the "
                f"field of view), {n_angular} spokes (a quarter of the data's), and "
                f"{n_radial} radial samples (the most whose frequencies stay inside "
                "the data's radius). Fewer samples transform faster.",
                id="suggestion",
                markup=False,
            )
        yield Label("The samples sit")
        yield _enum_select(
            options=((kind.value, kind) for kind in InputGrid),
            value=InputGrid.UNIFORM_POLAR if uniform else InputGrid.ON_GRID,
            select_id="input-grid",
        )
        yield Label("R: the grid's space limit (or band limit)")
        yield Input(value=R, placeholder="e.g. 1.0", id="R")
        yield Label("Limit kind")
        yield _enum_select(
            options=((kind.name, kind) for kind in LimitKind),
            value=LimitKind.SPACE_LIMITED,
            select_id="limit-kind",
        )
        yield Label(
            "Uniform polar input only: the grid's radial and angular sample counts, "
            "and the radius the samples cover (in units of R; for frequency data, "
            "in radians per unit of R)"
        )
        with Horizontal():
            yield Input(value=str(n_radial), id="n-radial", disabled=not uniform)
            yield Input(value=str(n_angular), id="n-angular", disabled=not uniform)
        radius = "" if loaded.radius is None else repr(loaded.radius)
        yield Input(
            value=radius, placeholder="radius", id="radius", disabled=not uniform
        )
        yield Label("Optional: a band limit, to check the Nyquist condition")
        yield Input(placeholder="band limit", id="band-limit")
        yield Button(label="Next", id="next", variant="primary")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Enable the uniform-only fields exactly when the input is uniform polar.

        :param event: The selection change.
        :type event: Select.Changed

        """
        if event.select.id == "input-grid":
            uniform = event.value is InputGrid.UNIFORM_POLAR
            for field_id in self._UNIFORM_FIELDS:
                self.query_one(selector=f"#{field_id}", expect_type=Input).disabled = (
                    not uniform
                )

    def _settings(self) -> GridSettings:
        """Read the fields into grid settings.

        :returns: The settings.
        :rtype: pypft.cli._pipeline.GridSettings
        :raises ValueError: If a field holds an invalid value.

        """
        input_grid = self._choice(select_id="input-grid", enum_class=InputGrid)
        R = parse_positive_float(text=self._field(field_id="R"), name="R")
        limit_kind = self._choice(select_id="limit-kind", enum_class=LimitKind)
        band_limit = parse_optional_positive_float(
            text=self._field(field_id="band-limit"), name="the band limit"
        )
        if input_grid is InputGrid.ON_GRID:
            # Samples on the grid fix its size: the array's own shape.
            n_radial, n_angular = self._loaded().values.shape[:2]
            radius = None
        else:
            n_radial = parse_positive_int(
                text=self._field(field_id="n-radial"), name="n_radial"
            )
            n_angular = parse_positive_int(
                text=self._field(field_id="n-angular"), name="n_angular"
            )
            radius = parse_positive_float(
                text=self._field(field_id="radius"), name="the radius"
            )
        return GridSettings(
            input_grid=input_grid,
            n_radial=n_radial,
            n_angular=n_angular,
            R=R,
            limit_kind=limit_kind,
            radius=radius,
            band_limit=band_limit,
        )

    def act(self, button_id: str) -> None:
        """Build and check the grid, then go on to the direction.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "next":
            settings = self._settings()
            grid, notices = check_grid(settings=settings)
            self.session.settings, self.session.grid_notices = settings, notices
            self.report(message=_lines([f"Grid: {grid}"], notices))
            self.app.push_screen(screen=DirectionScreen())


class DirectionScreen(StepScreen):
    """Choose the transform, the path, and whether to trace it, then run it."""

    HEADING = "The transform"

    def __init__(self) -> None:
        """Start with an empty status line and no notices."""
        super().__init__()
        self._notices_text = ""

    @property
    def notices(self) -> str:
        """What the screen says about the grid: its size and every check's warning.

        :returns: The text.
        :rtype: str

        """
        return self._notices_text

    def compose_step(self) -> ComposeResult:
        """Yield the grid's summary and the transform's choices.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        settings = self.session.settings
        summary = "" if settings is None else f"Grid: {settings.grid()}"
        notices = self.session.grid_notices or ("The grid passed its checks.",)
        self._notices_text = _lines([summary], notices)
        yield Static(self._notices_text, id="notices", markup=False)
        loaded = self.session.loaded
        # A .npz or acquisition file is k-space: frequency-domain input for the IPFT.
        inverse = loaded is not None and loaded.radius is not None
        yield Label("Transform")
        yield _enum_select(
            options=_DIRECTION_LABELS,
            value=Direction.INVERSE if inverse else Direction.FORWARD,
            select_id="direction",
        )
        yield Label("Path: exact (the default, exactly invertible) or accurate")
        yield _enum_select(
            options=((route.value, route) for route in Route),
            value=Route.EXACT,
            select_id="route",
        )
        yield Checkbox(label="Trace every domain (exact path only)", id="trace")
        yield Button(label="Transform", id="run", variant="primary")

    def _transform(self, direction: Direction, route: Route, trace: bool) -> None:
        """Run the transform and go on to the result, reporting any error on screen.

        :param direction: The PFT or the IPFT.
        :type direction: pypft.transform.Direction
        :param route: The path.
        :type route: pypft.cli._pipeline.Route
        :param trace: Whether to record every domain.
        :type trace: bool

        """
        loaded, settings = self.session.loaded, self.session.settings
        try:
            if loaded is None or settings is None:
                raise ValueError("read a file and describe its grid first")
            result = run_transform(
                loaded=loaded,
                settings=settings,
                direction=direction,
                route=route,
                trace=trace,
            )
        except _REPORTED_ERRORS as error:
            self.report(message=f"Error: {error}")
            return
        self.session.result = result
        self.report(message="")
        self.app.push_screen(screen=ResultScreen())

    def act(self, button_id: str) -> None:
        """Say the transform is running, then run it once that is drawn.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "run":
            direction = self._choice(select_id="direction", enum_class=Direction)
            route = self._choice(select_id="route", enum_class=Route)
            trace = self.query_one(selector="#trace", expect_type=Checkbox).value
            self.report(message=f"Transforming{_BUSY_SUFFIX}")
            self.call_after_refresh(
                self._transform, direction=direction, route=route, trace=trace
            )


# ======================================================================================
# The result
# ======================================================================================


class ConfirmScreen(ModalScreen[bool]):
    """Ask before replacing existing files.

    :param message: The question.
    :type message: str

    """

    def __init__(self, message: str) -> None:
        """Remember the question."""
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        """Lay out the question and its two answers.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        with Vertical(id="dialog"):
            yield Static(self._message, markup=False)
            with Horizontal():
                yield Button(label="Overwrite", id="yes", variant="error")
                yield Button(label="Cancel", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return the answer.

        :param event: The button press.
        :type event: Button.Pressed

        """
        self.dismiss(result=event.button.id == "yes")


class ResultScreen(StepScreen):
    """View the result's figures, one batch element at a time, or save them all."""

    HEADING = "The result"

    def __init__(self) -> None:
        """Start with no figure open."""
        super().__init__()
        self._figure: Figure | None = None
        self._interactive = False
        self._events: Timer | None = None
        self._view_directory: Path | None = None

    @property
    def figure(self) -> Figure | None:
        """The one figure every view draws into, once something was viewed.

        :returns: The figure, or ``None``.
        :rtype: Figure | None

        """
        return self._figure

    def _result(self) -> TransformResult:
        """Return the finished transform, or raise if nothing was transformed."""
        result = self.session.result
        if result is None:
            raise ValueError("nothing has been transformed")
        return result

    def _stem(self) -> str:
        """Return the input file's stem, which starts every saved file's name."""
        path = self.session.path
        if path is None:
            raise ValueError("no file has been read")
        return path.stem

    def compose_step(self) -> ComposeResult:
        """Yield the result's summary, the view controls and the save controls.

        :returns: The widgets.
        :rtype: ComposeResult

        """
        result = self._result()
        final = result.signals[-1]
        size = batch_size(signal=final)
        yield Static(
            _lines(
                [f"Result: {final.domain.name}, {final.values.shape}"],
                result.notices,
            ),
            id="summary",
            markup=False,
        )
        yield Label("Domain")
        yield _enum_select(
            options=((signal.domain.name, signal.domain) for signal in result.signals),
            value=final.domain,
            select_id="domain",
        )
        if size is not None:
            yield Label(f"Batch element, 0 .. {size - 1} (Enter shows it)")
            yield Input(value="0", id="batch-index")
        yield Button(label="View", id="view")
        yield Label("Save every figure into")
        path = self.session.path
        directory = (
            ""
            if path is None
            else str(path.parent / f"{path.stem}{_FIGURES_DIRECTORY_SUFFIX}")
        )
        yield Input(value=directory, id="directory")
        yield Button(label="Save", id="save", variant="primary")

    # ----------------------------------------------------------------------------------
    # Viewing
    # ----------------------------------------------------------------------------------

    def _selected(self) -> tuple[BaseSignal, int | None]:
        """Return the selected domain's signal at the selected batch element.

        :returns: The single sample, and its batch index (``None`` for a single
            sample).
        :rtype: tuple[pypft.domains.BaseSignal, int | None]

        """
        domain = self._choice(select_id="domain", enum_class=PolarDomain)
        signal = next(s for s in self._result().signals if s.domain is domain)
        if batch_size(signal=signal) is None:
            return signal, None
        index = self._field(field_id="batch-index").strip()
        try:
            batch_index = int(index)
        except ValueError:
            raise ValueError(
                f"the batch element must be an index, got {index!r}"
            ) from None
        return sample_of(signal=signal, batch_index=batch_index), batch_index

    def _flush_figure_events(self) -> None:
        """Let the open figure window redraw and respond."""
        if self._figure is not None:
            self._figure.canvas.flush_events()

    def _window_closed(self, _event: object) -> None:
        """Forget a figure whose window the user closed, so the next view reopens it."""
        self._figure = None

    def _open_figure(self) -> Figure:
        """Return the one view figure, opening it on first use.

        A figure whose window the user closed is opened again.

        :returns: The figure.
        :rtype: Figure

        """
        if self._figure is not None:
            return self._figure
        self._interactive = interactive_backend()
        figure = view_figure(interactive=self._interactive)
        if self._interactive:
            figure.canvas.mpl_connect(s="close_event", func=self._window_closed)
            figure.show()
            if self._events is None:
                self._events = self.set_interval(
                    interval=_FIGURE_EVENTS_INTERVAL,
                    callback=self._flush_figure_events,
                )
        self._figure = figure
        return figure

    def view(self) -> None:
        """Draw the selected sample into the one figure, opening it on first use.

        In a window when matplotlib's backend opens one; otherwise the figure is
        written to a temporary file whose path the status line shows.
        """
        sample, batch_index = self._selected()
        figure = self._open_figure()
        draw_signal(figure=figure, signal=sample)
        shown = sample.domain.name
        if batch_index is not None:
            shown = f"{shown}, batch element {batch_index},"
        if self._interactive:
            figure.canvas.draw_idle()
            self.report(message=f"Showing {shown} in the figure window")
            return
        if self._view_directory is None:
            self._view_directory = Path(tempfile.mkdtemp(prefix="pypft_view_"))
        path = self._view_directory / _VIEW_FILE_NAME
        figure.savefig(fname=path)
        self.report(
            message=f"No figure window on this backend; saved {shown} to {path}"
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Show another batch element in the same figure.

        :param event: The submitted field.
        :type event: Input.Submitted

        """
        if event.input.id == "batch-index":
            try:
                self.view()
            except _REPORTED_ERRORS as error:
                self.report(message=f"Error: {error}")

    def on_unmount(self) -> None:
        """Close the figure window, if one was opened."""
        if self._events is not None:
            self._events.stop()
        if self._figure is not None and self._interactive:
            plt.close(fig=self._figure)
        self._figure = None

    # ----------------------------------------------------------------------------------
    # Saving
    # ----------------------------------------------------------------------------------

    def _save(self, directory: Path, overwrite: bool, notices: list[str]) -> None:
        """Save every recorded domain's figures, reporting the outcome.

        :param directory: The directory to save into.
        :type directory: Path
        :param overwrite: Whether existing files may be replaced.
        :type overwrite: bool
        :param notices: What was shown before saving started, kept on screen.
        :type notices: list[str]

        """
        try:
            stem = self._stem()

            def save_all() -> list[Path]:
                return [
                    path
                    for signal in self._result().signals
                    for path in save_signal_figures(
                        signal=signal,
                        directory=directory,
                        stem=stem,
                        overwrite=overwrite,
                    )
                ]

            saved, messages = record_warnings(call=save_all)
        except _REPORTED_ERRORS as error:
            self.report(message=f"Error: {error}")
            return
        # The slowness warning is already among the notices shown before saving.
        others = [m for m in messages if not m.startswith(_SLOWNESS_PREFIX)]
        self.report(
            message=_lines(
                notices, others, [f"Saved {len(saved)} files to {directory}"]
            )
        )

    def _request_save(self, directory: Path) -> None:
        """Save, asking first when files would be replaced.

        :param directory: The directory to save into.
        :type directory: Path

        """

        def answer(overwrite: bool | None) -> None:
            if overwrite:
                self._start_save(directory=directory, overwrite=True)
            else:
                self.report(message="Nothing saved")

        existing = [
            path
            for signal in self._result().signals
            for path in figure_paths(
                signal=signal, directory=directory, stem=self._stem()
            )
            if path.exists()
        ]
        if not existing:
            self._start_save(directory=directory, overwrite=False)
            return
        self.app.push_screen(
            screen=ConfirmScreen(
                message=f"{len(existing)} file(s) exist in {directory}, e.g. "
                f"{existing[0].name}. Overwrite them?"
            ),
            callback=answer,
        )

    def _start_save(self, directory: Path, overwrite: bool) -> None:
        """Warn about a large batch on screen first, then save once that is drawn.

        :param directory: The directory to save into.
        :type directory: Path
        :param overwrite: Whether existing files may be replaced.
        :type overwrite: bool

        """
        size = batch_size(signal=self._result().signals[-1])
        _, messages = record_warnings(call=lambda: warn_if_slow(batch_size=size))
        self.report(message=_lines(messages, [f"Saving{_BUSY_SUFFIX}"]))
        self.call_after_refresh(
            self._save, directory=directory, overwrite=overwrite, notices=messages
        )

    def act(self, button_id: str) -> None:
        """View the selected sample, or save every figure.

        :param button_id: The pressed button's id.
        :type button_id: str

        """
        if button_id == "view":
            self.view()
        elif button_id == "save":
            text = self._field(field_id="directory").strip()
            if not text:
                raise ValueError("enter the directory to save into")
            self._request_save(directory=Path(text))


# ======================================================================================
# The app
# ======================================================================================


class PyPFTApp(App[int]):
    """PyPFT's terminal interface: load a polar array, transform it, view or save it.

    ``run`` returns ``0`` when the user quits.
    """

    TITLE = "PyPFT"

    CSS = """
    Screen { background: $surface; }
    Horizontal { height: auto; }
    Input { width: 1fr; }
    Button { min-width: 16; margin: 1 2 0 0; }

    #heading {
        width: 100%;
        padding: 1 2;
        background: $primary-background;
        text-style: bold;
    }
    #body { padding: 0 2 1 2; }
    #body Label { width: 1fr; margin: 1 0 0 0; color: $text-muted; }
    #body Checkbox { margin: 1 0 0 0; }

    StartScreen #body { align: center middle; }
    #choices { width: auto; height: auto; }
    #choices Button { width: 32; margin: 1 0 0 0; }

    #navigation { padding: 1 2 0 2; border-top: solid $primary; }
    #navigation Button { margin: 0 2 0 0; }
    #status { height: auto; max-height: 12; padding: 1 2; }

    ModalScreen { align: center middle; background: $background 60%; }
    #dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    #tree { height: 20; margin: 1 0 0 0; }
    """

    def __init__(self) -> None:
        """Start with nothing chosen."""
        super().__init__()
        self.session = Session()

    @property
    def status(self) -> str:
        """The current screen's status line.

        :returns: The message, or ``""`` on a screen without one.
        :rtype: str

        """
        screen = self.screen
        return screen.status if isinstance(screen, StepScreen) else ""

    def on_mount(self) -> None:
        """Open the start screen."""
        self.push_screen(screen=StartScreen())
