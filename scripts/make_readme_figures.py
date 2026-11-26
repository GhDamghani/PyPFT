"""Draw the figures shown in ``README.md``.

Writes, by default into ``docs/_static/readme/``:

- ``pipeline_light.svg``/``pipeline_dark.svg``: the PFT/IPFT chain through its
  four domains, one file per GitHub color theme.
- ``cardiac.png``: measured radial cardiac k-space (one coil, log magnitude)
  next to the image ``pypft.inverse_pft_ring`` reconstructs from every coil.
  The data is the MD-CNN excerpt ``pypft.samples`` downloads (Harvard
  Dataverse, doi:10.7910/DVN/CI3WB6, CC0 1.0).
- ``maze_pipeline.png``: ``tests/samples/maze_polar.tif`` traced through the
  four domains of ``pypft.forward_pft_traced``.

The README references the files by absolute URL on the ``main`` branch, so that
they also render on PyPI. Run from the repository root, after downloading the
test samples (``python -m pypft.samples --directory tests/samples``).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use(backend="Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import pypft  # noqa: E402
from pypft.acquisition import (  # noqa: E402
    RadialLineLayout,
    load_stored_array,
    select_radial_kspace,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_OUTPUT_DIRECTORY = Path("docs/_static/readme")
SAMPLE_DIRECTORY = Path("tests/samples")
CARDIAC_SAMPLE = SAMPLE_DIRECTORY / "P01_slc1_phs1_lins196_cols416_cha4.mat"
MAZE_SAMPLE = SAMPLE_DIRECTORY / "maze_polar.tif"

# The grids notebooks 09 and 08 use for the same data
CARDIAC_GRID = pypft.PolarGrid(n_radial=128, n_angular=98, R=208.0)
MAZE_GRID = pypft.PolarGrid(n_radial=576, n_angular=48, R=1.0)

DPI = 150
RENDER_SIZE = 400
KSPACE_DYNAMIC_RANGE = 5.0
# Brighten the reconstruction's darker tissue without clipping the blood pool
CARDIAC_GAMMA = 0.6
MAZE_GAMMA = 0.3
MAZE_FREQUENCY_DECADES = 4.0
FREQUENCY_DOMAINS = (
    pypft.PolarDomain.FREQUENCY_HARMONIC,
    pypft.PolarDomain.FREQUENCY_ANGULAR,
)

THEMES = {
    "light": {"text": "#1f2328", "box": "#f6f8fa", "edge": "#57606a"},
    "dark": {"text": "#e6edf3", "box": "#161b22", "edge": "#8b949e"},
}
DOMAINS = (
    ("$f(r, \\theta)$", "image"),
    ("$f_n(r)$", "spatial harmonics"),
    ("$F_n(\\rho)$", "frequency harmonics"),
    ("$F(\\rho, \\psi)$", "k-space"),
)
STEPS = ("angular FFT", "Hankel $H_n$\nper harmonic", "inverse\nangular FFT")


# =============================================================================
# Figures
# =============================================================================


def draw_pipeline(theme: str) -> Figure:
    """Draw the four domains joined by the three transform steps.

    :param theme: A key of ``THEMES``.
    :type theme: str
    :returns: The figure, with a transparent background.
    :rtype: Figure
    """
    colors = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 1.9))
    fig.patch.set_alpha(alpha=0.0)
    ax.set_axis_off()
    ax.set_xlim(left=0, right=len(DOMAINS) * 2.6 - 0.6)
    ax.set_ylim(bottom=-0.9, top=0.9)
    for index, (symbol, name) in enumerate(DOMAINS):
        x = index * 2.6 + 1.0
        ax.text(
            x=x,
            y=0.15,
            s=symbol,
            ha="center",
            va="center",
            fontsize=17,
            color=colors["text"],
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": colors["box"],
                "edgecolor": colors["edge"],
            },
        )
        ax.text(x=x, y=-0.65, s=name, ha="center", fontsize=10, color=colors["edge"])
    for index, step in enumerate(STEPS):
        left = index * 2.6 + 1.75
        ax.annotate(
            text="",
            xy=(left + 1.1, 0.15),
            xytext=(left, 0.15),
            arrowprops={"arrowstyle": "<|-|>", "color": colors["edge"]},
        )
        ax.text(
            x=left + 0.55,
            y=0.4,
            s=step,
            ha="center",
            va="bottom",
            fontsize=9,
            color=colors["text"],
        )
    return fig


def draw_cardiac() -> Figure:
    """Draw measured cardiac k-space next to its ring-route reconstruction.

    :returns: The figure.
    :rtype: Figure
    """
    lines = load_stored_array(path=CARDIAC_SAMPLE, name="data")
    batch = select_radial_kspace(
        values=lines,
        layout=RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX,
        selection={"coil": None},
    )
    n_radial, n_angular = batch.values.shape[:2]

    # Coil 0's spokes as a disk, out to the radius every spoke reaches: the
    # asymmetric echo leaves the outer part of half the spokes empty
    spokes = batch.values[..., 0]
    n_shown = int(np.min(a=np.sum(a=np.abs(spokes) > 0, axis=0)))
    rho = np.arange(n_shown + 1) * batch.radius / n_radial
    psi = (np.arange(n_angular + 1) - n_angular // 2 - 0.5) * 2 * np.pi / n_angular
    magnitude = np.log10(np.abs(spokes[:n_shown]) + 1e-12)

    # Every coil reconstructed, combined by root sum of squares
    f = pypft.inverse_pft_ring(F=batch.values, grid=CARDIAC_GRID, radius=batch.radius)
    image = np.sqrt(np.sum(a=np.abs(f) ** 2, axis=-1))

    fig, (kspace_ax, image_ax) = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5), facecolor="black"
    )
    kspace_ax.pcolormesh(
        rho[:, np.newaxis] * np.cos(psi),
        rho[:, np.newaxis] * np.sin(psi),
        magnitude,
        vmin=magnitude.max() - KSPACE_DYNAMIC_RANGE,
        cmap="magma",
    )
    kspace_ax.set_aspect(aspect="equal")
    kspace_ax.invert_yaxis()
    signal = pypft.PolarSpatialAngularSignal(
        values=image**CARDIAC_GAMMA, grid=CARDIAC_GRID
    )
    pypft.render_cartesian(
        signal=signal, height=RENDER_SIZE, width=RENDER_SIZE, ax=image_ax
    )
    for ax, title in zip(
        (kspace_ax, image_ax),
        ("Measured radial k-space", "Reconstructed with PyPFT"),
    ):
        ax.set_axis_off()
        ax.set_title(label=title, color="white", fontsize=14)
    fig.tight_layout()
    return fig


def draw_maze() -> Figure:
    """Trace the maze fixture through the four domains of the forward PFT.

    The two physical-angle domains are drawn as disks, the two harmonic
    domains as ``(radial, harmonic)`` tables; every panel shows a
    gamma-enhanced magnitude.

    :returns: The figure.
    :rtype: Figure
    """
    values = plt.imread(fname=MAZE_SAMPLE).astype(np.float64)
    trace = pypft.forward_pft_traced(f=values, grid=MAZE_GRID)

    fig, axes = plt.subplots(nrows=1, ncols=len(DOMAINS), figsize=(14, 3.8))
    for ax, signal, (symbol, name) in zip(axes, trace.signals, DOMAINS):
        magnitude = np.abs(signal.values)
        if signal.domain in FREQUENCY_DOMAINS:
            # Log scale over MAZE_FREQUENCY_DECADES below the peak
            relative = np.log10(magnitude / magnitude.max() + 1e-300)
            magnitude = np.clip(a=relative, a_min=-MAZE_FREQUENCY_DECADES, a_max=0)
            magnitude = magnitude + MAZE_FREQUENCY_DECADES
        else:
            magnitude = magnitude**MAZE_GAMMA
        if signal.domain in (
            pypft.PolarDomain.SPATIAL_ANGULAR,
            pypft.PolarDomain.FREQUENCY_ANGULAR,
        ):
            shown = type(signal)(values=magnitude, grid=MAZE_GRID)
            pypft.render_cartesian(
                signal=shown, height=RENDER_SIZE, width=RENDER_SIZE, ax=ax
            )
        else:
            ax.imshow(X=magnitude, aspect="auto", origin="lower")
        ax.set_axis_off()
        ax.set_title(label=f"{symbol}\n{name}", fontsize=13)
    fig.tight_layout()
    return fig


# =============================================================================
# Command line
# =============================================================================


def main() -> None:
    """Write every README figure into ``--output-dir``."""
    parser = argparse.ArgumentParser(description="Draw the figures shown in README.md.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    output_directory: Path = parser.parse_args().output_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    for theme in THEMES:
        fig = draw_pipeline(theme=theme)
        # svg.hashsalt keeps the SVG's element ids stable between runs
        with plt.rc_context(rc={"svg.hashsalt": "pypft", "svg.fonttype": "path"}):
            fig.savefig(
                fname=output_directory / f"pipeline_{theme}.svg",
                bbox_inches="tight",
                metadata={"Date": None},
            )
        plt.close(fig=fig)

    for name, draw in (("cardiac", draw_cardiac), ("maze_pipeline", draw_maze)):
        fig = draw()
        fig.savefig(
            fname=output_directory / f"{name}.png",
            dpi=DPI,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            metadata={"Software": None},
        )
        plt.close(fig=fig)
        print(f"wrote {output_directory / name}.png")


if __name__ == "__main__":
    main()
