from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pypft.io.artifacts import build_field_figure_path
from pypft.visualization._matplotlib_helpers import (
    axis_labels,
    load_matplotlib,
    render_payload,
)
from pypft.visualization.arrays import magnitude, phase, squeeze_single_sample
from pypft.visualization.models import ComplexView, FieldRenderSpec, RenderView


def save_field_comparison(
    reference_frame: Any,
    candidate_frame: Any,
    output_dir: Path,
    *,
    render_spec: FieldRenderSpec,
    stem: str = "comparison",
) -> tuple[Path, ...]:
    reference = _frame_array(reference_frame)
    candidate = _frame_array(candidate_frame)
    if reference.shape != candidate.shape:
        raise ValueError(
            "Comparison plots require reference and candidate arrays to "
            f"share one shape. Got {reference.shape} and {candidate.shape}."
        )

    field_kind = getattr(reference_frame, "field_kind", "field")
    reference_label = _frame_label(reference_frame, fallback="reference")
    candidate_label = _frame_label(candidate_frame, fallback="candidate")

    saved: list[Path] = []
    for view in _resolve_views(
        reference,
        candidate,
        render_spec.complex_view,
    ):
        path = build_field_figure_path(output_dir, stem=stem, view=view)
        _save_comparison_figure(
            reference,
            candidate,
            field_kind=field_kind,
            path=path,
            view=view,
            gamma=render_spec.gamma,
            reference_label=reference_label,
            candidate_label=candidate_label,
        )
        saved.append(path)

    return tuple(saved)


def _frame_array(frame: Any) -> np.ndarray:
    values = squeeze_single_sample(frame.asarray(copy=False))
    if values.ndim != 2:
        raise ValueError("Comparison rendering expects 2D arrays.")
    return values


def _frame_label(frame: Any, *, fallback: str) -> str:
    stage = getattr(frame, "stage", fallback)
    return str(stage).replace("_", " ").title()


def _resolve_views(
    reference: np.ndarray,
    candidate: np.ndarray,
    complex_view: ComplexView,
) -> tuple[RenderView, ...]:
    if not np.iscomplexobj(reference) and not np.iscomplexobj(candidate):
        return ("real",)
    if complex_view == "both":
        return ("magnitude", "phase")
    return (complex_view,)


def _save_comparison_figure(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    field_kind: str,
    path: Path,
    view: RenderView,
    gamma: float,
    reference_label: str,
    candidate_label: str,
) -> None:
    plt = load_matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True,
    )
    panels = (
        (reference_label, *render_payload(reference, view=view, gamma=gamma)),
        (candidate_label, *render_payload(candidate, view=view, gamma=gamma)),
        ("Difference", *_difference_payload(reference, candidate, view=view)),
    )

    x_label, y_label = axis_labels(_FieldKindProxy(field_kind=field_kind))
    for axes_item, (title, image, cmap, colorbar_label, color_limits) in zip(
        axes,
        panels,
        strict=True,
    ):
        plot = axes_item.imshow(
            image,
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )
        if color_limits is not None:
            plot.set_clim(*color_limits)
        axes_item.set_title(title)
        axes_item.set_xlabel(x_label)
        axes_item.set_ylabel(y_label)
        figure.colorbar(plot, ax=axes_item, label=colorbar_label)

    figure.suptitle(f"{view.title()} Comparison", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _difference_payload(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    view: RenderView,
) -> tuple[np.ndarray, str, str, tuple[float, float] | None]:
    if view == "real":
        difference = np.real(candidate) - np.real(reference)
        return _signed_payload(difference, label="difference")
    if view == "magnitude":
        difference = np.abs(magnitude(candidate) - magnitude(reference))
        return difference, "magma", "abs magnitude error", None

    phase_delta = np.angle(np.exp(1j * (phase(candidate) - phase(reference))))
    return phase_delta, "twilight", "phase delta [rad]", (-np.pi, np.pi)


def _signed_payload(
    values: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, str, str, tuple[float, float] | None]:
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    if data_min < 0.0 < data_max:
        limit = max(abs(data_min), abs(data_max))
        return values, "RdBu_r", label, (-limit, limit)
    return values, "viridis", label, None


class _FieldKindProxy:
    def __init__(self, *, field_kind: str) -> None:
        self.field_kind = field_kind


__all__ = ["save_field_comparison"]
