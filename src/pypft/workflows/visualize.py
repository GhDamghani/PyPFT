from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from pypft import PyPFT
from pypft.core.exceptions import MetadataValidationError
from pypft.io import (
    load_array,
    load_transform_run_manifest,
)
from pypft.tracing import (
    ReplayFrame,
    TransformTrace,
    load_saved_trace,
    load_stored_trace_document,
)
from pypft.visualization import (
    FieldRenderSpec,
    normalize_complex_view,
    save_field_figures,
    save_trace_figures,
    squeeze_single_sample,
)

TraceSourceKind = Literal["manifest", "trace"]


@dataclass(frozen=True, slots=True)
class TraceInspectionFrame:
    index: int
    stage: str
    field_kind: str
    shape: tuple[int, ...]
    dtype: str
    has_array: bool
    figure_count: int


@dataclass(frozen=True, slots=True)
class TraceInspection:
    source_kind: TraceSourceKind
    source_path: Path
    direction: str
    trace_path: Path
    frames: tuple[TraceInspectionFrame, ...]
    input_path: Path | None = None
    metadata_path: Path | None = None
    backend: str | None = None
    dht_implementation: str | None = None
    gamma: float | None = None
    complex_view: str | None = None
    package_version: str | None = None


def replay_trace_from_manifest(manifest_path: Path) -> TransformTrace:
    """Re-run a transform from a saved manifest.

    This replay uses the current input file referenced by the manifest, so the
    caller should treat the result as valid only when that file still exists
    and has not changed since the original run.
    """

    manifest = load_transform_run_manifest(manifest_path)
    input_path = manifest.resolve_input_path(manifest_path=manifest_path)
    values = load_array(input_path)
    if values.ndim != 2:
        raise MetadataValidationError(
            "Manifest replay requires one 2D sample with shape (n_r, n_theta)."
        )

    expected_shape = (
        manifest.metadata.spatial_grid.shape
        if manifest.direction == "forward"
        else manifest.metadata.frequency_grid.shape
    )
    if values.shape != expected_shape:
        raise MetadataValidationError(
            f"Manifest metadata shape {expected_shape!r} does not match the "
            f"current input shape {values.shape!r}."
        )

    pft = PyPFT(
        backend=manifest.backend,
        dht_implementation=manifest.dht_implementation,
        spatial_grid=manifest.metadata.spatial_grid,
        frequency_grid=manifest.metadata.frequency_grid,
    )
    if manifest.direction == "forward":
        return pft.forward_with_trace(values).trace
    return pft.backward_with_trace(values).trace


def render_trace_source(
    source_path: Path,
    output_dir: Path,
    *,
    gamma: float | None = None,
    complex_view: str | None = None,
) -> dict[str, tuple[Path, ...]]:
    source_kind = _detect_trace_source_kind(source_path)
    if source_kind == "manifest":
        manifest = load_transform_run_manifest(source_path)
        trace = replay_trace_from_manifest(source_path)
        render_spec = FieldRenderSpec(
            complex_view=(
                normalize_complex_view(complex_view)
                if complex_view is not None
                else manifest.complex_view
            ),
            gamma=manifest.gamma if gamma is None else gamma,
        )
        return save_trace_figures(
            trace,
            output_dir,
            render_spec=render_spec,
            save_all_views=True,
        )

    render_spec = FieldRenderSpec(
        complex_view=(
            normalize_complex_view(complex_view)
            if complex_view is not None
            else "both"
        ),
        gamma=1.0 if gamma is None else gamma,
    )
    return save_trace_figures(
        load_saved_trace(source_path),
        output_dir,
        render_spec=render_spec,
        save_all_views=True,
    )


def render_field_file(
    input_path: Path,
    output_dir: Path,
    *,
    field_kind: str,
    direction: str = "unknown",
    gamma: float,
    complex_view: str,
) -> tuple[Path, ...]:
    values = squeeze_single_sample(load_array(input_path))
    if values.ndim != 2:
        raise MetadataValidationError(
            "Field visualization expects a 2D array or a single-sample batch."
        )

    frame = ReplayFrame(
        stage=field_kind,
        direction=direction,
        field_kind=field_kind,
        values=values,
        grid={},
    )
    return save_field_figures(
        frame,
        output_dir,
        render_spec=FieldRenderSpec(
            complex_view=normalize_complex_view(complex_view),
            gamma=gamma,
        ),
        stem=f"{input_path.stem}.{field_kind}",
    )


def inspect_trace_source(source_path: Path) -> TraceInspection:
    source_kind = _detect_trace_source_kind(source_path)
    if source_kind == "manifest":
        manifest = load_transform_run_manifest(source_path)
        trace_path = manifest.resolve_trace_path(manifest_path=source_path)
        trace_document = load_stored_trace_document(trace_path)
        return TraceInspection(
            source_kind="manifest",
            source_path=source_path,
            direction=manifest.direction,
            trace_path=trace_path,
            frames=_inspection_frames(trace_document),
            input_path=manifest.resolve_input_path(manifest_path=source_path),
            metadata_path=manifest.resolve_metadata_path(
                manifest_path=source_path,
            ),
            backend=manifest.backend,
            dht_implementation=manifest.dht_implementation,
            gamma=manifest.gamma,
            complex_view=manifest.complex_view,
            package_version=manifest.package_version,
        )

    trace_document = load_stored_trace_document(source_path)
    return TraceInspection(
        source_kind="trace",
        source_path=source_path,
        direction=trace_document.direction,
        trace_path=source_path,
        frames=_inspection_frames(trace_document),
    )


def _inspection_frames(trace_document) -> tuple[TraceInspectionFrame, ...]:
    return tuple(
        TraceInspectionFrame(
            index=frame.index,
            stage=frame.stage,
            field_kind=frame.field_kind,
            shape=frame.shape,
            dtype=frame.dtype,
            has_array=frame.array_path is not None,
            figure_count=len(frame.figure_paths),
        )
        for frame in trace_document.frames
    )


def _detect_trace_source_kind(source_path: Path) -> TraceSourceKind:
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Artifact file {source_path} is not valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise MetadataValidationError("Artifact JSON must be an object.")
    if "frames" in payload:
        return "trace"
    if "metadata" in payload and "trace_path" in payload:
        return "manifest"
    raise MetadataValidationError(
        "Artifact file must be either a transform manifest or a trace "
        "document."
    )


__all__ = [
    "TraceInspection",
    "TraceInspectionFrame",
    "inspect_trace_source",
    "render_field_file",
    "render_trace_source",
    "replay_trace_from_manifest",
]
