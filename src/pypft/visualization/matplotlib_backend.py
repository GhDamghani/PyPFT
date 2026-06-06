from __future__ import annotations

from pathlib import Path

import numpy as np

from pypft.core.exceptions import VisualizationDependencyError
from pypft.tracing import TraceFrame
from pypft.visualization.arrays import (
    apply_gamma,
    magnitude,
    phase,
    squeeze_single_sample,
)
from pypft.visualization.models import RenderView


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise VisualizationDependencyError(
            "Static figure rendering requires the optional matplotlib "
            "dependency. Install PyPFT with the viz extra."
        ) from error
    return plt


class MatplotlibFieldRenderer:
    def save_frame(
        self,
        frame: TraceFrame,
        path: Path,
        *,
        view: RenderView,
        gamma: float,
    ) -> None:
        plt = _load_matplotlib()
        data = squeeze_single_sample(frame.asarray(copy=False))
        if data.ndim != 2:
            raise ValueError("MatplotlibFieldRenderer expects a 2D field.")

        figure, axes = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image, cmap, colorbar_label = _render_payload(
            data,
            view=view,
            gamma=gamma,
        )
        plot = axes.imshow(image, aspect="auto", origin="lower", cmap=cmap)
        if view == "phase":
            plot.set_clim(-np.pi, np.pi)
        axes.set_title(_format_title(frame, view=view, gamma=gamma))
        x_label, y_label = _axis_labels(frame)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        figure.colorbar(plot, ax=axes, label=colorbar_label)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)


def _render_payload(
    data: np.ndarray,
    *,
    view: RenderView,
    gamma: float,
) -> tuple[np.ndarray, str, str]:
    if view == "real":
        return np.real(data), "viridis", "value"
    if view == "magnitude":
        return apply_gamma(magnitude(data), gamma), "viridis", "magnitude"
    return phase(data), "twilight", "phase [rad]"


def _format_title(frame: TraceFrame, *, view: RenderView, gamma: float) -> str:
    stage_label = frame.stage.replace("_", " ").title()
    view_label = view.title()
    if view == "magnitude" and gamma != 1.0:
        return f"{stage_label} ({view_label}, gamma={gamma:g})"
    return f"{stage_label} ({view_label})"


def _axis_labels(frame: TraceFrame) -> tuple[str, str]:
    if frame.field_kind == "spatial_samples":
        return ("angular index", "radial index")
    if frame.field_kind == "angular_spectrum":
        return ("angular order", "radial index")
    if frame.field_kind == "radial_spectrum":
        return ("angular order", "radial frequency index")
    return ("angular frequency index", "radial frequency index")


__all__ = ["MatplotlibFieldRenderer"]