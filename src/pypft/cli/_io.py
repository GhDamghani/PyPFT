"""The terminal interface's file handling: loading arrays, naming and saving figures.

Everything here is independent of the user interface, so it is tested directly and
the app only calls it.

**Input files** hold a polar array in PyPFT's ``(radial, angular[, batch])`` layout,
with a centered angular axis (``POLAR_FILE_SUFFIXES``, read with ``load_array``):

- ``.npy``: a single sample or a batch, real or complex (``numpy.save``).
- ``.npz``: uniform polar k-space saved by ``pypft.acquisition.RadialKSpace.save``,
  a single sample or a batch, carrying the frequency radius its samples cover.
- ``.tif``/``.tiff``/``.png``: an 8- or 16-bit grayscale image holding a single
  real sample, read with ``matplotlib.pyplot.imread`` (which scales a PNG to
  ``[0, 1]`` and keeps a TIFF's integer values).

or radial k-space lines in one of ``pypft.acquisition.RadialLineLayout``'s layouts
(``LINE_FILE_SUFFIXES``), stored in any file ``pypft.acquisition.load_stored_array``
reads: a ``.npy`` file's single array, or one named array of a ``.npz`` or ``.mat``
file. ``pypft.acquisition.selectable_axes`` lists the axes the lines are recorded
over, and ``acquisition_input`` converts a selection of one index per axis (a single
sample) or of whole axes (a batch) into uniform polar k-space, as from a ``.npz`` file
of ``RadialKSpace.save``.

**Output figures** are ``.png`` files named
``<stem>_<domain>_<magnitude|phase>[_b<NNN>].png``: the input file's stem, the domain
slug (``polar_`` and the lower-cased ``pypft.domains.PolarDomain`` member, as in
``pypft.viz``'s trace labels), the part of ``pypft.viz.plot_signal``'s pair the file
shows, and for a batch the zero-padded batch index, three digits wide or as wide as
the largest index needs.
"""

import warnings
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypft.acquisition import (
    ARRAY_FILE_SUFFIXES,
    RadialKSpace,
    RadialLineLayout,
    select_radial_kspace,
)
from pypft.acquisition._kspace import _KSPACE_SUFFIX
from pypft.axes import POLAR_SAMPLE_NDIM
from pypft.domains import BaseSignal, PolarDomain, _type_is_base_signal
from pypft.utils.validators import (
    EnumValidator,
    IntValidator,
    MatplotlibValidator,
    NumpyValidator,
    PathValidator,
    StrValidator,
)
from pypft.viz import _domain_label, plot_signal

# ======================================================================================
# Constants
# ======================================================================================

#: What a call whose warnings are recorded returns.
T = TypeVar("T")

#: The suffix of a NumPy array file: any polar array, real or complex.
ARRAY_SUFFIX = ".npy"

#: The suffixes of an image file: a single real sample.
IMAGE_SUFFIXES = (".tif", ".tiff", ".png")

#: The suffixes of a polar array file, read with ``load_array``.
POLAR_FILE_SUFFIXES = (ARRAY_SUFFIX, _KSPACE_SUFFIX, *IMAGE_SUFFIXES)

#: The suffixes of a file of radial k-space lines, read with
#: ``pypft.acquisition.load_stored_array``.
LINE_FILE_SUFFIXES = ARRAY_FILE_SUFFIXES

#: Every suffix the terminal interface reads, each once.
SUPPORTED_SUFFIXES = tuple(dict.fromkeys((*POLAR_FILE_SUFFIXES, *LINE_FILE_SUFFIXES)))

#: The suffix of every saved figure.
FIGURE_SUFFIX = ".png"

#: The largest batch saved without a ``CLIPerformanceWarning``: every element is
#: rendered into two figures of its own, which takes a noticeable time per element.
SLOW_BATCH_SIZE = 32

#: The narrowest batch index in a file name; wider when the batch needs more digits.
_MIN_BATCH_INDEX_WIDTH = 3

#: What the readers raise on an empty or corrupt file, besides ``OSError`` and
#: ``ValueError``: NumPy on a truncated file (``EOFError``) or a broken archive
#: (``zipfile.BadZipFile``), and Pillow on a broken PNG (``SyntaxError``).
_UNREADABLE_FILE_ERRORS = (EOFError, zipfile.BadZipFile, SyntaxError)

#: The array kinds a polar array may hold: bool, integers, floats, complex numbers.
_NUMERIC_KINDS = "biufc"

#: matplotlib's built-in backends that never open a window (lower-cased); an inline
#: notebook backend draws into the notebook instead, so it never opens one either.
_NON_INTERACTIVE_BACKENDS = frozenset(
    {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
)
_INLINE_BACKEND_MARKER = "inline"

#: The size of a saved figure, in inches.
_FIGURE_SIZE = (5.0, 4.0)
_VIEW_FIGURE_SIZE = (10.0, 4.0)


# ======================================================================================
# Types
# ======================================================================================


class CLIPerformanceWarning(UserWarning):
    """Warns that saving every element of a large batch may be slow."""


class FigurePart(Enum):
    """The two images ``pypft.viz.plot_signal`` draws, saved to separate files."""

    MAGNITUDE = "magnitude"
    PHASE = "phase"


@dataclass(frozen=True, eq=False)
class LoadedInput:
    """A polar array read from a file, with what the file says about it.

    Not comparable with ``==`` (``eq=False``), since its ``values`` is an array.

    :param values: The array, ``(radial, angular)`` or ``(radial, angular, batch)``.
    :type values: np.ndarray
    :param radius: The frequency radius a ``.npz`` file's or an acquisition's uniform
        polar k-space covers, in radians per pixel; ``None`` for every other file.
    :type radius: float | None
    :param is_image: Whether the array was read from an image file, so it is real and
        holds a spatial-domain sample.
    :type is_image: bool

    """

    values: np.ndarray
    radius: float | None
    is_image: bool


def _type_is_loaded_input(value: LoadedInput) -> None:
    """Type-validator for ``LoadedInput``.

    :param value: The value to be validated.
    :type value: LoadedInput
    :raises TypeError: If the value is not a ``LoadedInput``.

    """
    if not isinstance(value, LoadedInput):
        raise TypeError(
            f"value must be {LoadedInput.__name__}, got {type(value).__name__}"
        )


# ======================================================================================
# Warnings
# ======================================================================================


def record_warnings(call: Callable[[], T]) -> tuple[T, list[str]]:
    """Run ``call``, recording every warning it emits instead of letting it escape.

    A terminal interface owns the screen, so a warning printed to it would be lost;
    the app shows the recorded text instead.

    :param call: What to run.
    :type call: Callable[[], T]
    :returns: The call's result and every recorded warning as
        ``"<category>: <message>"``, in order.
    :rtype: tuple[T, list[str]]

    """
    with warnings.catch_warnings(record=True) as recorded:
        # "always": a warning already shown once elsewhere is recorded here too.
        warnings.simplefilter(action="always")
        result = call()
    return result, [f"{item.category.__name__}: {item.message}" for item in recorded]


def warn_if_slow(batch_size: int | None) -> None:
    """Warn that saving a batch this large renders many figures, one at a time.

    :param batch_size: The batch size, or ``None`` for a single sample.
    :type batch_size: int | None
    :raises TypeError: If ``batch_size`` is neither an ``int`` nor ``None``.

    """
    if batch_size is None:
        return
    IntValidator.type_is_int(value=batch_size)
    if batch_size > SLOW_BATCH_SIZE:
        warnings.warn(
            message=(
                f"saving {batch_size} batch elements renders {2 * batch_size} "
                "figures, one at a time; this may be slow"
            ),
            category=CLIPerformanceWarning,
            stacklevel=2,
        )


# ======================================================================================
# Loading
# ======================================================================================


def _read_image(path: Path, *, batched: bool) -> np.ndarray:
    """Read a grayscale image as one real sample.

    :param path: The image file.
    :type path: Path
    :param batched: Whether a batch was asked for.
    :type batched: bool
    :returns: The image as a ``float64`` ``(radial, angular)`` array.
    :rtype: np.ndarray
    :raises ValueError: If a batch was asked for, or the image is not grayscale.

    """
    if batched:
        raise ValueError(
            f"{path.name} is an image, which holds a single sample; save a batch "
            f"as a {ARRAY_SUFFIX} file"
        )
    image = plt.imread(fname=path)
    if image.ndim != POLAR_SAMPLE_NDIM:
        raise ValueError(
            f"{path.name} has color channels {image.shape}; a polar image must be "
            "grayscale, (radial, angular)"
        )
    return np.asarray(image, dtype=np.float64)


def load_array(path: Path, *, batched: bool) -> LoadedInput:
    """Read a polar array from a file and check it against the chosen layout.

    :param path: A ``.npy``, ``.npz``, ``.tif``, ``.tiff`` or ``.png`` file; see this
        module's docstring for what each holds. A file of radial k-space lines is
        read with ``pypft.acquisition.load_stored_array`` and ``acquisition_input``
        instead.
    :type path: Path
    :param batched: ``True`` for a batch ``(radial, angular, batch)``, ``False`` for
        a single sample ``(radial, angular)``.
    :type batched: bool
    :returns: The array, its stored radius (``.npz`` only), and whether it came from
        an image.
    :rtype: LoadedInput
    :raises TypeError: If ``path`` is not a ``pathlib.Path`` or ``batched`` not a
        ``bool``.
    :raises ValueError: If the suffix is not a polar array file's, the file is empty
        or corrupt
        or holds no numeric array, the array does not match ``batched``, or it has a
        non-finite element.
    :raises OSError: If the file does not exist, is a directory, or cannot be read.

    """
    PathValidator.type_is_Path(value=path)
    if not isinstance(batched, bool):
        raise TypeError(f"batched must be bool, got {type(batched).__name__}")
    if path.suffix.lower() not in POLAR_FILE_SUFFIXES and (
        path.suffix.lower() in LINE_FILE_SUFFIXES
    ):
        raise ValueError(
            f"{path.name} is read only as radial k-space lines: choose a layout of "
            "radial lines"
        )
    PathValidator.value_has_suffix_in(value=path, suffixes=POLAR_FILE_SUFFIXES)
    PathValidator.value_is_a_file(value=path)
    PathValidator.value_is_readable(value=path)
    suffix = path.suffix.lower()
    radius = None
    try:
        if suffix == ARRAY_SUFFIX:
            # allow_pickle=False: a file is read as a plain array, never as code.
            # The file is opened here so it is closed whatever np.load raises.
            with path.open(mode="rb") as file:
                values = np.load(file=file, allow_pickle=False)
        elif suffix == _KSPACE_SUFFIX:
            # np.load leaves a broken archive's file open, so it is checked first.
            if not zipfile.is_zipfile(filename=path):
                raise ValueError(f"{path.name} is not a {_KSPACE_SUFFIX} archive")
            kspace = RadialKSpace.load(path=path)
            values, radius = kspace.values, kspace.radius
        else:
            values = _read_image(path=path, batched=batched)
    except _UNREADABLE_FILE_ERRORS as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error
    # np.load reads an archive whatever its suffix, as a mapping of arrays.
    if not isinstance(values, np.ndarray):
        raise ValueError(f"{path.name} holds an archive, not one {ARRAY_SUFFIX} array")
    if values.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"{path.name} holds {values.dtype} values, not numbers")
    _check_layout(values=values, name=path.name, batched=batched)
    return LoadedInput(values=values, radius=radius, is_image=suffix in IMAGE_SUFFIXES)


def _check_layout(values: np.ndarray, name: str, *, batched: bool) -> None:
    """Check an array read from a file against the choice of a sample or a batch.

    :param values: The array.
    :type values: np.ndarray
    :param name: The file's name, for the error message.
    :type name: str
    :param batched: Whether a batch was asked for.
    :type batched: bool
    :raises ValueError: If the array's rank does not match ``batched``, or it has a
        non-finite element.

    """
    # The rank the choice of a single sample or a batch implies.
    expected = POLAR_SAMPLE_NDIM + 1 if batched else POLAR_SAMPLE_NDIM
    if values.ndim != expected:
        layout = "(radial, angular, batch)" if batched else "(radial, angular)"
        raise ValueError(
            f"{name} holds a {values.ndim}-dimensional array {values.shape}; "
            f"expected {layout}"
        )
    NumpyValidator.value_is_finite(value=values)


def acquisition_input(
    values: np.ndarray,
    *,
    layout: RadialLineLayout,
    selection: Mapping[str, int | None],
) -> LoadedInput:
    """Convert a selection of an array of radial lines into uniform polar k-space.

    The selection works as in ``pypft.acquisition.select_radial_kspace``: one index
    per selectable axis, or ``None`` to keep the whole axis in the batch. With every
    axis at one index the result is a single sample, otherwise a batch.

    :param values: The array of radial lines, as
        ``pypft.acquisition.load_stored_array`` reads it from its file.
    :type values: np.ndarray
    :param layout: The array's layout of radial lines.
    :type layout: pypft.acquisition.RadialLineLayout
    :param selection: The index, or ``None``, per selectable axis.
    :type selection: Mapping[str, int | None]
    :returns: The k-space and the frequency radius it covers; never an image.
    :rtype: LoadedInput
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If the selection names an unknown axis or an index outside
        its axis, the array is not in the layout, or the k-space has a non-finite
        element.

    """
    kspace = select_radial_kspace(values=values, layout=layout, selection=selection)
    NumpyValidator.value_is_finite(value=kspace.values)
    return LoadedInput(values=kspace.values, radius=kspace.radius, is_image=False)


# ======================================================================================
# Naming
# ======================================================================================


def batch_size(signal: BaseSignal) -> int | None:
    """Count a signal's batch elements.

    :param signal: The signal.
    :type signal: pypft.domains.BaseSignal
    :returns: The batch size, or ``None`` for a single sample.
    :rtype: int | None
    :raises TypeError: If ``signal`` is not a ``pypft.domains.BaseSignal``.

    """
    _type_is_base_signal(value=signal)
    if signal.values.ndim == POLAR_SAMPLE_NDIM:
        return None
    return signal.values.shape[signal.batch_axis]


def sample_of(signal: BaseSignal, batch_index: int | None) -> BaseSignal:
    """Select one sample of a signal, in the signal's own domain.

    :param signal: The signal, a single sample or a batch.
    :type signal: pypft.domains.BaseSignal
    :param batch_index: The batch element; ``None`` for a single sample.
    :type batch_index: int | None
    :returns: ``signal`` itself for a single sample, else the element as a signal of
        the same class on the same grid.
    :rtype: pypft.domains.BaseSignal
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``batch_index`` is given for a single sample, missing for
        a batch, or outside the batch.

    """
    size = batch_size(signal=signal)
    if size is None:
        if batch_index is not None:
            raise ValueError("a single sample has no batch index")
        return signal
    if batch_index is None:
        raise ValueError("a batch needs a batch index")
    IntValidator.type_is_int(value=batch_index)
    IntValidator.value_is_non_negative(value=batch_index)
    IntValidator.value1_is_less_than_value2(value1=batch_index, value2=size)
    values = np.take(a=signal.values, indices=batch_index, axis=signal.batch_axis)
    return type(signal)(values=values, grid=signal.grid)


def output_filename(
    stem: str,
    domain: PolarDomain,
    part: FigurePart,
    batch_index: int | None = None,
    batch_size: int | None = None,
) -> str:
    """Name one saved figure: ``<stem>_<domain>_<part>[_b<NNN>].png``.

    :param stem: The input file's stem.
    :type stem: str
    :param domain: The domain the figure shows.
    :type domain: pypft.domains.PolarDomain
    :param part: Which image of the magnitude/phase pair the figure shows.
    :type part: FigurePart
    :param batch_index: The batch element shown; ``None`` for a single sample.
    :type batch_index: int | None
    :param batch_size: The batch's size, which sets the index's width; ``None`` for a
        single sample.
    :type batch_size: int | None
    :returns: The file name.
    :rtype: str
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``stem`` is empty, only one of ``batch_index`` and
        ``batch_size`` is given, or ``batch_index`` is outside the batch.

    """
    StrValidator.type_is_str(value=stem)
    if not stem:
        raise ValueError("stem must not be empty")
    EnumValidator.type_is_enum(value=domain)
    EnumValidator.value_is_enum_member(value=domain, enum_class=PolarDomain)
    EnumValidator.type_is_enum(value=part)
    EnumValidator.value_is_enum_member(value=part, enum_class=FigurePart)
    name = f"{stem}_{_domain_label(domain=domain)}_{part.value}"
    if batch_index is None and batch_size is None:
        return f"{name}{FIGURE_SUFFIX}"
    if batch_index is None or batch_size is None:
        raise ValueError("batch_index and batch_size are given together or not at all")
    IntValidator.type_is_int(value=batch_index)
    IntValidator.type_is_int(value=batch_size)
    IntValidator.value_is_non_negative(value=batch_index)
    IntValidator.value1_is_less_than_value2(value1=batch_index, value2=batch_size)
    # Every index of one batch gets the same width, so the files sort in order.
    width = max(_MIN_BATCH_INDEX_WIDTH, len(str(batch_size - 1)))
    return f"{name}_b{batch_index:0{width}d}{FIGURE_SUFFIX}"


# ======================================================================================
# Rendering
# ======================================================================================


def interactive_backend() -> bool:
    """Tell whether matplotlib's current backend opens figure windows.

    :returns: ``False`` for a file-only backend (e.g. ``Agg``) or an inline notebook
        backend, else ``True``.
    :rtype: bool

    """
    backend = matplotlib.get_backend().lower()
    return not (
        backend in _NON_INTERACTIVE_BACKENDS or _INLINE_BACKEND_MARKER in backend
    )


def draw_signal(figure: Figure, signal: BaseSignal) -> tuple[Axes, Axes]:
    """Draw a single sample into a figure, replacing whatever the figure held.

    The figure is cleared and redrawn in place, so one figure shows every batch
    element in turn.

    :param figure: The figure to draw into.
    :type figure: Figure
    :param signal: A single sample.
    :type signal: pypft.domains.BaseSignal
    :returns: The ``(magnitude_ax, phase_ax)`` pair drawn.
    :rtype: tuple[Axes, Axes]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``signal`` is a batch.

    """
    MatplotlibValidator.type_is_figure(value=figure)
    figure.clear()
    magnitude_ax = figure.add_subplot(1, 2, 1)
    phase_ax = figure.add_subplot(1, 2, 2)
    return plot_signal(signal=signal, ax=(magnitude_ax, phase_ax))


def _part_figure(signal: BaseSignal, part: FigurePart) -> Figure:
    """Render one image of a single sample's magnitude/phase pair on its own figure.

    ``plot_signal`` always draws both images, so the other one goes to an axes on a
    second, discarded figure. Both figures are created without ``pyplot``, so none is
    registered with it and nothing is left open.

    :param signal: A single sample.
    :type signal: pypft.domains.BaseSignal
    :param part: The image to keep.
    :type part: FigurePart
    :returns: The figure holding that image.
    :rtype: Figure

    """
    figure = Figure(figsize=_FIGURE_SIZE, layout="constrained")
    kept = figure.add_subplot()
    discarded = Figure().add_subplot()
    if part is FigurePart.MAGNITUDE:
        plot_signal(signal=signal, ax=(kept, discarded))
    else:
        plot_signal(signal=signal, ax=(discarded, kept))
    return figure


def view_figure(*, interactive: bool) -> Figure:
    """Create the one figure a result is viewed in.

    :param interactive: Whether to open it as a ``pyplot`` window (closed with
        ``matplotlib.pyplot.close``) or create it off screen, unregistered.
    :type interactive: bool
    :returns: The figure.
    :rtype: Figure

    """
    if interactive:
        return plt.figure(figsize=_VIEW_FIGURE_SIZE, layout="constrained")
    return Figure(figsize=_VIEW_FIGURE_SIZE, layout="constrained")


# ======================================================================================
# Saving
# ======================================================================================


def figure_paths(signal: BaseSignal, directory: Path, *, stem: str) -> tuple[Path, ...]:
    """List the files ``save_signal_figures`` writes for a signal, in order.

    :param signal: The signal, a single sample or a batch.
    :type signal: pypft.domains.BaseSignal
    :param directory: The directory the files go to.
    :type directory: Path
    :param stem: The input file's stem.
    :type stem: str
    :returns: One path per batch element and part, magnitude before phase.
    :rtype: tuple[Path, ...]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``stem`` is empty.

    """
    PathValidator.type_is_Path(value=directory)
    size = batch_size(signal=signal)
    indices: list[int | None] = [None] if size is None else list(range(size))
    return tuple(
        directory
        / output_filename(
            stem=stem,
            domain=signal.domain,
            part=part,
            batch_index=index,
            batch_size=size,
        )
        for index in indices
        for part in FigurePart
    )


def save_signal_figures(
    signal: BaseSignal, directory: Path, *, stem: str, overwrite: bool = False
) -> tuple[Path, ...]:
    """Save a signal's magnitude and phase, per batch element, as ``.png`` files.

    The directory is created when missing. Above ``SLOW_BATCH_SIZE`` elements a
    ``CLIPerformanceWarning`` is emitted before anything is rendered.

    :param signal: The signal, a single sample or a batch.
    :type signal: pypft.domains.BaseSignal
    :param directory: The directory to save into.
    :type directory: Path
    :param stem: The input file's stem, the start of every file name.
    :type stem: str
    :param overwrite: Whether existing files may be replaced.
    :type overwrite: bool
    :returns: The saved files, in ``figure_paths`` order.
    :rtype: tuple[Path, ...]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``stem`` is empty.
    :raises FileExistsError: If a file exists and ``overwrite`` is ``False``; nothing
        is written then.
    :raises OSError: If ``directory`` cannot be created or written.

    """
    paths = figure_paths(signal=signal, directory=directory, stem=stem)
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be bool, got {type(overwrite).__name__}")
    existing = [path.name for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} file(s) exist, e.g. {existing[0]}")
    PathValidator.value_is_writable(value=directory)
    size = batch_size(signal=signal)
    warn_if_slow(batch_size=size)
    directory.mkdir(parents=True, exist_ok=True)
    indices: list[int | None] = [None] if size is None else list(range(size))
    # paths runs over the same (index, part) pairs in the same order.
    targets = iter(paths)
    for index in indices:
        sample = sample_of(signal=signal, batch_index=index)
        for part in FigurePart:
            figure = _part_figure(signal=sample, part=part)
            figure.savefig(fname=next(targets))
    return paths
