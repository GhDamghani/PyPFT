"""Figures for a polar signal, as ``Axes``/``Figure`` objects.

``plot_signal`` renders one ``pypft.domains.BaseSignal`` at a time as a
(magnitude, phase) pair of images -- a gamma-enhanced magnitude
(``matplotlib.colors.PowerNorm``, never a hand-rolled ``**gamma``) and a phase
map. Every domain is assumed complex-valued, ``SPACE_POLAR`` and
``SPACE_HARMONIC`` included: a full forward-then-inverse round trip can leave
even a ``SPACE_POLAR`` signal with a non-trivial phase, and ``SPACE_HARMONIC``
is an angular DFT's own coefficients -- generically complex even when the
space-domain signal being transformed is real (a DFT of real input is only
symmetric, not real, in general). ``SPACE_POLAR``'s own magnitude is
additionally rendered in grayscale rather than ``matplotlib``'s default
colormap, everywhere it is drawn -- both ``plot_signal`` and
``render_cartesian`` -- since it is literally a photographic image (see
``pypft.grid.sample_cartesian``), unlike every other domain's more abstract
magnitude. ``render_cartesian`` is the
display-only counterpart for the two ``POLAR`` domains, whose angular axis is a
physical angle rather than a harmonic order: it interpolates
``pypft.grid.PolarGrid``'s own non-uniform sample points onto an ordinary Cartesian
pixel grid via ``scipy.interpolate.griddata``, purely so a polar-domain signal can be
shown next to an ordinary image -- its output is an approximation, never fed back
into ``pypft.transform.forward_pft``/``inverse_pft``.

``forward_pft_traced``/``inverse_pft_traced`` run the same three-step chain as
``forward_pft``/``inverse_pft`` (by walking ``pypft.domains.BaseSignal``'s own
verified chain, rather than duplicating the pipeline), optionally building one
figure per domain (``visualize_steps``) and/or one holistic mosaic of every
domain at once (``visualize_pipeline``). Every combination returns the same
``PFTTrace`` type, since a return type that depends on an argument's *value*
rather than its *type* is a ``pyright`` defect -- this is why tracing is a
separate entry point instead of a ``visualize=``/``record=`` flag on
``forward_pft``/``inverse_pft`` themselves.

Neither ``plot_signal``/``render_cartesian`` nor ``forward_pft_traced``/
``inverse_pft_traced`` write to disk on their own -- every figure stays open
in memory until the caller closes it (``PFTTrace.close()`` for a traced call, or
``matplotlib.pyplot.close`` directly for a bare ``plot_signal``/``render_cartesian``
call). ``PFTTrace.save`` is the one explicit, opt-in exception: called directly by
the caller, never implicitly by tracing itself, it writes every figure in the
trace to a directory as enumerated PNGs, mirroring the naming a user would
otherwise have hand-rolled themselves (``00_...``, ``01_...``, ...). This matters
for figure lifecycle because ``pytest.ini_options: filterwarnings = ["error"]``
turns matplotlib's own "more than 20 figures have been opened" ``RuntimeWarning``
into a hard test failure, so any caller (tests included) that creates many figures
without closing them will eventually fail the suite even though nothing about the
figures themselves is wrong.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from pypft.axes import DEFAULT_BATCH_AXIS
from pypft.domains import (
    BaseSignal,
    Domain,
    FrequencyPolarSignal,
    SpacePolarSignal,
    _type_is_base_signal,
)
from pypft.grid import PolarGrid, _type_is_polar_grid
from pypft.utils.validators import (
    IntValidator,
    MatplotlibValidator,
    NumpyValidator,
    PathValidator,
)

#: The two ``Domain`` members whose angular axis is a physical angle rather than a
#: harmonic order -- the only ones ``render_cartesian`` can meaningfully interpolate
#: onto a Cartesian grid.
_POLAR_DOMAINS = (Domain.SPACE_POLAR, Domain.FREQUENCY_POLAR)

#: Gamma applied to every magnitude plot via ``matplotlib.colors.PowerNorm`` --
#: compresses the large dynamic range typical of a complex-valued signal's
#: magnitude so structure away from the peak stays visible.
_MAGNITUDE_GAMMA = 0.3

#: The magnitude colormap for ``Domain.SPACE_POLAR`` specifically -- unlike every
#: other domain's magnitude (an abstract Fourier/harmonic coefficient, left at
#: ``matplotlib``'s own default colormap), a ``SPACE_POLAR`` magnitude is literally a
#: photographic image (see ``pypft.grid.sample_cartesian``'s own image argument), so
#: it renders in grayscale to look like one.
_SPACE_POLAR_MAGNITUDE_CMAP = "gray"

#: The fixed color range for every phase plot -- ``np.angle``'s own output range,
#: pinned explicitly rather than left to ``matplotlib``'s auto-scaling. See
#: ``DESIGN_NOTES.md``, "Visualization: phase color range is pinned to
#: ``[-pi, pi]``," for why (a degenerate-``Normalize`` failure mode for any
#: exactly-constant phase).
_PHASE_VMIN = -np.pi
_PHASE_VMAX = np.pi


def _magnitude_cmap(domain: Domain) -> str | None:
    """Resolve the magnitude colormap for ``domain`` -- grayscale for ``SPACE_POLAR``.

    :param domain: The signal's own domain.
    :type domain: pypft.domains.Domain
    :returns: ``_SPACE_POLAR_MAGNITUDE_CMAP`` for ``Domain.SPACE_POLAR``, else
        ``None`` (``matplotlib``'s own default colormap).
    :rtype: str | None

    """
    return _SPACE_POLAR_MAGNITUDE_CMAP if domain is Domain.SPACE_POLAR else None


# ======================================================================================
# Plotting a single signal
# ======================================================================================


def _new_axes() -> Axes:
    """Create a fresh, single ``Axes`` on its own ``Figure``."""
    _, ax = plt.subplots()
    return ax


def plot_signal(
    signal: BaseSignal, ax: tuple[Axes, Axes] | None = None
) -> tuple[Axes, Axes]:
    """Plot a signal as a (magnitude, phase) pair of images.

    Every domain is assumed complex-valued -- see this module's own docstring
    for why neither ``Domain.SPACE_POLAR`` nor ``Domain.SPACE_HARMONIC`` is an
    exception, despite both nominally being "space-domain."

    :param signal: The signal to render. Must be 2-D (no batch axis).
    :type signal: pypft.domains.BaseSignal
    :param ax: The ``(magnitude_ax, phase_ax)`` pair to render onto, or
        ``None`` to create both.
    :type ax: tuple[Axes, Axes] | None
    :returns: The ``(magnitude_ax, phase_ax)`` pair used.
    :rtype: tuple[Axes, Axes]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``signal.values`` is not 2-D.

    """
    _type_is_base_signal(value=signal)
    NumpyValidator.value_is_2d(value=signal.values)
    if ax is None:
        _, (magnitude_ax, phase_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    else:
        if not (isinstance(ax, tuple) and len(ax) == 2):
            if isinstance(ax, tuple):
                raise TypeError(
                    f"ax must be a tuple of two Axes, got a tuple of length {len(ax)}"
                )
            raise TypeError(f"ax must be a tuple of two Axes, got {type(ax).__name__}")
        magnitude_ax, phase_ax = ax
        MatplotlibValidator.type_is_axes(value=magnitude_ax)
        MatplotlibValidator.type_is_axes(value=phase_ax)
    magnitude = np.abs(signal.values)
    # PowerNorm (not a hand-rolled `magnitude ** gamma`) keeps the colorbar's own
    # tick values meaningful in the original units.
    magnitude_ax.imshow(
        magnitude,
        aspect="auto",
        origin="lower",
        cmap=_magnitude_cmap(domain=signal.domain),
        norm=PowerNorm(gamma=_MAGNITUDE_GAMMA, vmin=0.0, vmax=magnitude.max()),
    )
    magnitude_ax.set_xlabel("angular index")
    magnitude_ax.set_ylabel("radial index")
    magnitude_ax.set_title(f"{signal.domain.name} magnitude")
    phase_ax.imshow(
        np.angle(signal.values),
        aspect="auto",
        origin="lower",
        cmap="twilight",
        vmin=_PHASE_VMIN,
        vmax=_PHASE_VMAX,
    )
    phase_ax.set_xlabel("angular index")
    phase_ax.set_ylabel("radial index")
    phase_ax.set_title(f"{signal.domain.name} phase")
    return magnitude_ax, phase_ax


# ======================================================================================
# Cartesian rendering, for display only
# ======================================================================================


def _polar_sample_points(grid: PolarGrid) -> np.ndarray:
    """Convert ``grid``'s own polar sample points to Cartesian ``(x, y)`` coordinates.

    ``grid.theta`` follows ``pypft.geometry``'s own image-coordinate
    convention -- measured directly on image coordinates, so increasing angle
    rotates toward increasing row (downward on screen), not toward standard math
    "up". Since ``imshow``'s own ``origin="lower"`` instead expects ``y`` to
    increase upward, ``sin(theta)`` must be negated here or a reconstructed image
    comes out upside down.

    :param grid: The grid whose sample points to convert.
    :type grid: pypft.grid.PolarGrid
    :returns: An ``(n_angular * n_radial, 2)`` array of ``(x, y)`` points.
    :rtype: np.ndarray

    """
    x = grid.r * np.cos(grid.theta)[:, np.newaxis]
    y = -grid.r * np.sin(grid.theta)[:, np.newaxis]
    return np.column_stack((x.ravel(), y.ravel()))


def render_cartesian(
    signal: BaseSignal, *, height: int, width: int, ax: Axes | None = None
) -> Axes:
    """Interpolate a polar-domain signal onto a Cartesian grid, for display only.

    Uses ``scipy.interpolate.griddata`` on ``signal.grid``'s own non-uniform,
    order-dependent sample points (``pypft.grid.PolarGrid.r``/``.theta`` -- the
    same points ``pypft.grid.sample_cartesian`` samples *from*), an
    approximation reconstructed purely so a polar-domain signal can be shown
    next to an ordinary image. **Its output must never be fed back into
    ``pypft.transform.forward_pft``/``inverse_pft``**: interpolating onto a
    uniform Cartesian grid and back would not reproduce the original samples,
    unlike the exact transform chain.

    :param signal: The polar-domain signal to render -- must be in
        ``Domain.SPACE_POLAR`` or ``Domain.FREQUENCY_POLAR``, the two domains
        whose angular axis is a physical angle rather than a harmonic order.
        Must be 2-D (no batch axis).
    :type signal: pypft.domains.BaseSignal
    :param height: The rendered image's height, in pixels.
    :type height: int
    :param width: The rendered image's width, in pixels.
    :type width: int
    :param ax: The ``Axes`` to render onto, or ``None`` to create one.
    :type ax: Axes | None
    :returns: ``ax``, with the interpolated Cartesian magnitude image drawn
        onto it.
    :rtype: Axes
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``signal.domain`` is not one of the two polar
        domains, ``signal.values`` is not 2-D, or ``height``/``width`` is not
        positive.

    """
    _type_is_base_signal(value=signal)
    if signal.domain not in _POLAR_DOMAINS:
        raise ValueError(
            f"signal.domain must be SPACE_POLAR or FREQUENCY_POLAR (a physical "
            f"angle axis), got {signal.domain.name}"
        )
    NumpyValidator.value_is_2d(value=signal.values)
    IntValidator.type_is_int(value=height)
    IntValidator.value_is_positive(value=height)
    IntValidator.type_is_int(value=width)
    IntValidator.value_is_positive(value=width)
    if ax is None:
        ax = _new_axes()
    else:
        MatplotlibValidator.type_is_axes(value=ax)

    grid = signal.grid
    # signal.values is (n_radial, n_angular); grid.r/.theta are
    # (n_angular, n_radial)/(n_angular,) -- transpose to align both onto the
    # same layout before flattening into a scattered point cloud for griddata.
    points = _polar_sample_points(grid=grid)
    magnitude = np.abs(signal.values).T.ravel()

    xi = np.linspace(-grid.R, grid.R, width)
    yi = np.linspace(-grid.R, grid.R, height)
    grid_x, grid_y = np.meshgrid(xi, yi)
    cartesian = griddata(points, magnitude, (grid_x, grid_y), method="linear")

    ax.imshow(
        cartesian,
        extent=(-grid.R, grid.R, -grid.R, grid.R),
        origin="lower",
        cmap=_magnitude_cmap(domain=signal.domain),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{signal.domain.name} (Cartesian, interpolated)")
    return ax


# ======================================================================================
# Traced forward/inverse PFT
# ======================================================================================


def _pipeline_mosaic(signals: tuple[BaseSignal, ...]) -> Figure:
    """Render every domain a traced call passed through onto one holistic figure.

    :param signals: Every domain a traced PFT/IPFT call passed through, in
        pipeline order.
    :type signals: tuple[pypft.domains.BaseSignal, ...]
    :returns: One figure with two (magnitude, phase) panels per domain.
    :rtype: Figure

    """
    total_panels = 2 * len(signals)
    fig, axes = plt.subplots(nrows=1, ncols=total_panels, figsize=(4 * total_panels, 4))
    for index, signal in enumerate(signals):
        plot_signal(signal=signal, ax=(axes[2 * index], axes[2 * index + 1]))
    fig.suptitle("PFT pipeline")
    fig.tight_layout()
    return fig


def _step_figure(signal: BaseSignal) -> Figure:
    """Render one domain's own step figure.

    :param signal: The signal to render this step's figure for.
    :type signal: pypft.domains.BaseSignal
    :returns: The one figure built for this domain step.
    :rtype: Figure

    """
    fig, (magnitude_ax, phase_ax) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    plot_signal(signal=signal, ax=(magnitude_ax, phase_ax))
    fig.tight_layout()
    return fig


def _trace_figures(
    signals: tuple[BaseSignal, ...],
    *,
    visualize_steps: bool,
    visualize_pipeline: bool,
) -> tuple[tuple[Figure, ...], tuple[str, ...]]:
    """Build the figures a traced call was asked for, plus a matching label per figure.

    :param signals: Every domain a traced PFT/IPFT call passed through, in
        pipeline order.
    :type signals: tuple[pypft.domains.BaseSignal, ...]
    :param visualize_steps: Whether to render one figure per domain.
    :type visualize_steps: bool
    :param visualize_pipeline: Whether to render one holistic mosaic figure.
    :type visualize_pipeline: bool
    :returns: The figures created and a same-length tuple of labels naming
        each one (``"step_<domain>"`` or ``"pipeline"``), in the order: the
        steps (if any), then the pipeline mosaic (if any).
    :rtype: tuple[tuple[Figure, ...], tuple[str, ...]]

    """
    figures: list[Figure] = []
    labels: list[str] = []
    if visualize_steps:
        for signal in signals:
            figures.append(_step_figure(signal=signal))
            labels.append(f"step_{signal.domain.name.lower()}")
    if visualize_pipeline:
        figures.append(_pipeline_mosaic(signals=signals))
        labels.append("pipeline")
    return tuple(figures), tuple(labels)


@dataclass(frozen=True)
class PFTTrace:
    """The result of a traced forward or inverse PFT, with optional figures.

    :param values: The final transform output -- identical to what
        ``pypft.transform.forward_pft``/``inverse_pft`` would return.
    :type values: np.ndarray
    :param signals: Every domain the trace passed through, in pipeline order
        (4 signals: the start domain, two intermediates, and the end domain).
    :type signals: tuple[pypft.domains.BaseSignal, ...]
    :param figures: Every ``Figure`` created for this trace -- empty unless
        ``visualize_steps``/``visualize_pipeline`` was requested. Marked
        ``compare=False``/``repr=False`` since a ``Figure`` is neither
        comparable nor usefully representable.
    :type figures: tuple[Figure, ...]
    :param figure_labels: One label per element of ``figures``, in the same
        order -- what ``save`` names each file after.
    :type figure_labels: tuple[str, ...]
    :raises TypeError: If any element of ``figures`` is not a ``Figure``.
    :raises ValueError: If ``figures`` and ``figure_labels`` have different
        lengths.

    """

    values: np.ndarray
    signals: tuple[BaseSignal, ...]
    figures: tuple[Figure, ...] = field(default=(), compare=False, repr=False)
    figure_labels: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        """Validate ``figures``/``figure_labels``, right after construction."""
        for figure in self.figures:
            MatplotlibValidator.type_is_figure(value=figure)
        if len(self.figures) != len(self.figure_labels):
            raise ValueError(
                f"figures has {len(self.figures)} entries but figure_labels "
                f"has {len(self.figure_labels)}"
            )

    def close(self) -> None:
        """Close every figure this trace created.

        ``pytest.ini_options: filterwarnings = ["error"]`` turns matplotlib's
        own "more than 20 figures have been opened" ``RuntimeWarning`` into a
        test failure, so a trace's figures must be closed once no longer
        needed rather than left open indefinitely.
        """
        for figure in self.figures:
            plt.close(figure)

    def save(self, directory: Path) -> tuple[Path, ...]:
        """Save every figure in this trace to ``directory``, enumerated in order.

        The one explicit, opt-in place this module ever writes to disk --
        never called implicitly by ``forward_pft_traced``/``inverse_pft_traced``
        themselves. Each file is named ``"<index>_<label>.png"`` (e.g.
        ``"00_step_space_polar.png"``), using this trace's own
        ``figure_labels``.

        :param directory: Where to save each figure -- created, along with any
            missing parent directories, if it does not already exist.
        :type directory: Path
        :returns: The path written for each figure, in the same order as
            ``figures``.
        :rtype: tuple[Path, ...]
        :raises TypeError: If ``directory`` is not a ``Path``.
        :raises FileNotFoundError: If no existing ancestor directory of
            ``directory`` can be found on the filesystem.
        :raises NotADirectoryError: If ``directory``'s closest existing
            ancestor is not a directory.
        :raises PermissionError: If ``directory`` (or its closest existing
            ancestor) is not writable.

        """
        PathValidator.type_is_Path(value=directory)
        PathValidator.value_is_writable(value=directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, (figure, label) in enumerate(zip(self.figures, self.figure_labels)):
            path = directory / f"{index:02d}_{label}.png"
            figure.savefig(path)
            paths.append(path)
        return tuple(paths)


def forward_pft_traced(
    f: np.ndarray,
    grid: PolarGrid,
    *,
    batch_axis: int = DEFAULT_BATCH_AXIS,
    visualize_steps: bool = False,
    visualize_pipeline: bool = False,
) -> PFTTrace:
    """Compute the forward PFT, recording every domain it passes through.

    Walks ``pypft.domains.SpacePolarSignal``'s own verified chain
    (``to_harmonics`` -> ``to_frequency`` -> ``to_angles``) rather than
    duplicating ``pypft.transform.forward_pft``'s pipeline, so ``values`` is
    identical to calling ``forward_pft`` directly.

    :param f: The space-domain samples ``f(r, theta)``; see ``forward_pft``.
    :type f: np.ndarray
    :param grid: The sampling grid ``f`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``f`` holding the batch dimension, only
        meaningful for a 3-D ``f``.
    :type batch_axis: int
    :param visualize_steps: Whether to render one figure per domain.
    :type visualize_steps: bool
    :param visualize_pipeline: Whether to render one holistic mosaic figure
        of every domain at once.
    :type visualize_pipeline: bool
    :returns: The traced result -- always a ``PFTTrace``, regardless of which
        visualization keywords were set.
    :rtype: PFTTrace
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``f`` is not 2-D or 3-D, or its shape does not
        match ``grid``.

    """
    _type_is_polar_grid(value=grid)
    start = SpacePolarSignal(values=f, grid=grid, batch_axis=batch_axis)
    harmonic = start.to_harmonics()
    frequency = harmonic.to_frequency()
    end = frequency.to_angles()
    signals = (start, harmonic, frequency, end)
    figures, figure_labels = _trace_figures(
        signals=signals,
        visualize_steps=visualize_steps,
        visualize_pipeline=visualize_pipeline,
    )
    return PFTTrace(
        values=end.values, signals=signals, figures=figures, figure_labels=figure_labels
    )


def inverse_pft_traced(
    F: np.ndarray,
    grid: PolarGrid,
    *,
    batch_axis: int = DEFAULT_BATCH_AXIS,
    visualize_steps: bool = False,
    visualize_pipeline: bool = False,
) -> PFTTrace:
    """Compute the inverse PFT, recording every domain it passes through.

    The exact mirror of ``forward_pft_traced``: walks
    ``pypft.domains.FrequencyPolarSignal``'s own verified chain
    (``to_harmonics`` -> ``to_space`` -> ``to_angles``), so ``values`` is
    identical to calling ``pypft.transform.inverse_pft`` directly.

    :param F: The frequency-domain samples ``F(rho, phi)``; see
        ``inverse_pft``.
    :type F: np.ndarray
    :param grid: The sampling grid ``F`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``F`` holding the batch dimension, only
        meaningful for a 3-D ``F``.
    :type batch_axis: int
    :param visualize_steps: Whether to render one figure per domain.
    :type visualize_steps: bool
    :param visualize_pipeline: Whether to render one holistic mosaic figure
        of every domain at once.
    :type visualize_pipeline: bool
    :returns: The traced result -- always a ``PFTTrace``, regardless of which
        visualization keywords were set.
    :rtype: PFTTrace
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``F`` is not 2-D or 3-D, or its shape does not
        match ``grid``.

    """
    _type_is_polar_grid(value=grid)
    start = FrequencyPolarSignal(values=F, grid=grid, batch_axis=batch_axis)
    harmonic = start.to_harmonics()
    space = harmonic.to_space()
    end = space.to_angles()
    signals = (start, harmonic, space, end)
    figures, figure_labels = _trace_figures(
        signals=signals,
        visualize_steps=visualize_steps,
        visualize_pipeline=visualize_pipeline,
    )
    return PFTTrace(
        values=end.values, signals=signals, figures=figures, figure_labels=figure_labels
    )
