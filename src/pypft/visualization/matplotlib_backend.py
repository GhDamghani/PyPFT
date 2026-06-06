from __future__ import annotations

from pathlib import Path
from pypft.tracing import TraceFrame
from pypft.visualization._matplotlib_helpers import (
    axis_labels,
    load_matplotlib,
    render_payload,
)
from pypft.visualization.arrays import squeeze_single_sample
from pypft.visualization.models import RenderView


class MatplotlibFieldRenderer:
    def save_frame(
        self,
        frame: TraceFrame,
        path: Path,
        *,
        view: RenderView,
        gamma: float,
    ) -> None:
        plt = load_matplotlib()
        data = squeeze_single_sample(frame.asarray(copy=False))
        if data.ndim != 2:
            raise ValueError("MatplotlibFieldRenderer expects a 2D field.")

        figure, axes = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image, cmap, colorbar_label, color_limits = render_payload(
            data,
            view=view,
            gamma=gamma,
        )
        plot = axes.imshow(image, aspect="auto", origin="lower", cmap=cmap)
        if color_limits is not None:
            plot.set_clim(*color_limits)
        axes.set_title(_format_title(frame, view=view, gamma=gamma))
        x_label, y_label = axis_labels(frame)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        figure.colorbar(plot, ax=axes, label=colorbar_label)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)


def _format_title(frame: TraceFrame, *, view: RenderView, gamma: float) -> str:
    stage_label = frame.stage.replace("_", " ").title()
    view_label = view.title()
    if view == "magnitude" and gamma != 1.0:
        return f"{stage_label} ({view_label}, gamma={gamma:g})"
    return f"{stage_label} ({view_label})"


__all__ = ["MatplotlibFieldRenderer"]
