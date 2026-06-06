from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pypft.io.artifacts import build_field_figure_path, build_trace_figure_path
from pypft.tracing import TransformTrace
from pypft.visualization.arrays import squeeze_single_sample
from pypft.visualization.matplotlib_backend import MatplotlibFieldRenderer
from pypft.visualization.models import ComplexView, FieldRenderSpec, RenderView
from pypft.visualization.renderers import FieldRenderer


def save_trace_figures(
    trace: TransformTrace,
    output_dir: Path,
    *,
    render_spec: FieldRenderSpec,
    save_all_views: bool,
    renderer: FieldRenderer | None = None,
) -> dict[str, tuple[Path, ...]]:
    active_renderer = renderer or MatplotlibFieldRenderer()
    frames = trace.frames if save_all_views else (trace.final_frame,)
    saved: dict[str, tuple[Path, ...]] = {}

    for index, frame in enumerate(frames, start=1):
        paths_for_stage: list[Path] = []
        for view in _resolve_views(frame, render_spec.complex_view):
            path = build_trace_figure_path(
                output_dir,
                index=index,
                stage=frame.stage,
                view=view,
            )
            active_renderer.save_frame(
                frame,
                path,
                view=view,
                gamma=render_spec.gamma,
            )
            paths_for_stage.append(path)
        saved[frame.stage] = tuple(paths_for_stage)

    return saved


def save_field_figures(
    frame: Any,
    output_dir: Path,
    *,
    render_spec: FieldRenderSpec,
    stem: str | None = None,
    renderer: FieldRenderer | None = None,
) -> tuple[Path, ...]:
    active_renderer = renderer or MatplotlibFieldRenderer()
    figure_stem = stem or frame.stage
    saved: list[Path] = []

    for view in _resolve_views(frame, render_spec.complex_view):
        path = build_field_figure_path(output_dir, stem=figure_stem, view=view)
        active_renderer.save_frame(
            frame,
            path,
            view=view,
            gamma=render_spec.gamma,
        )
        saved.append(path)

    return tuple(saved)


def _resolve_views(
    frame: Any,
    complex_view: ComplexView,
) -> tuple[RenderView, ...]:
    data = squeeze_single_sample(frame.asarray(copy=False))
    if not np.iscomplexobj(data):
        return ("real",)
    if complex_view == "both":
        return ("magnitude", "phase")
    return (complex_view,)


__all__ = ["save_field_figures", "save_trace_figures"]

