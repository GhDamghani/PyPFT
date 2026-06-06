from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pypft
from pypft import PyPFT
from pypft.io import (
    TransformRunManifest,
    build_stage_array_path,
    load_spatial_sample,
    load_transform_metadata,
    prepare_transform_artifacts,
    relative_artifact_path,
    save_array,
    write_json,
)
from pypft.tracing import TracedTransformResult, TransformTrace
from pypft.visualization import (
    FieldRenderSpec,
    save_trace_figures,
    squeeze_single_sample,
)
from pypft.workflows.requests import TransformWorkflowRequest
from pypft.workflows.results import TransformWorkflowResult


def run_transform_workflow(
    request: TransformWorkflowRequest,
) -> TransformWorkflowResult:
    values = load_spatial_sample(request.input_path)
    expected_domain = (
        "spatial" if request.direction == "forward" else "frequency"
    )
    metadata, metadata_source = load_transform_metadata(
        request.input_path,
        expected_domain=expected_domain,
        runtime_shape=values.shape,
        metadata_path=request.metadata_path,
    )

    backend = request.backend or metadata.backend or "cpu"
    dht_implementation = (
        request.dht_implementation or metadata.dht_implementation or "naive"
    )
    pft = PyPFT(
        backend=backend,
        dht_implementation=dht_implementation,
        spatial_grid=metadata.spatial_grid,
        frequency_grid=metadata.frequency_grid,
    )
    traced = _run_traced_transform(pft, values, direction=request.direction)

    artifacts = prepare_transform_artifacts(request.output_dir)
    save_array(artifacts.output_array_path, _output_array(traced.output))
    stage_array_paths = _save_stage_arrays(
        traced.trace,
        artifacts.stage_array_dir,
        enabled=request.save_stage_arrays,
    )
    figure_paths: dict[str, tuple[Path, ...]] = {}
    if request.save_all_views:
        figure_paths = save_trace_figures(
            traced.trace,
            artifacts.figure_dir,
            render_spec=FieldRenderSpec(
                complex_view=request.complex_view,
                gamma=request.gamma,
            ),
            save_all_views=True,
        )

    trace_payload = _serialize_trace(
        traced.trace,
        artifact_root=artifacts.output_dir,
        stage_array_paths=stage_array_paths,
        figure_paths=figure_paths,
    )
    write_json(artifacts.trace_path, trace_payload)

    manifest = TransformRunManifest(
        schema_version=2,
        direction=request.direction,
        input_path=str(request.input_path),
        metadata_path=str(metadata_source),
        output_array_path=relative_artifact_path(
            artifacts.output_array_path,
            root=artifacts.output_dir,
        ),
        trace_path=relative_artifact_path(
            artifacts.trace_path,
            root=artifacts.output_dir,
        ),
        manifest_path=relative_artifact_path(
            artifacts.manifest_path,
            root=artifacts.output_dir,
        ),
        package_version=pypft.__version__,
        backend=backend,
        dht_implementation=dht_implementation,
        gamma=request.gamma,
        complex_view=request.complex_view,
        save_all_views=request.save_all_views,
        save_stage_arrays=request.save_stage_arrays,
        metadata=metadata,
        artifact_root=".",
        trace_stage_names=traced.trace.stage_names,
        stage_array_paths={
            stage: relative_artifact_path(path, root=artifacts.output_dir)
            for stage, path in stage_array_paths.items()
        },
        figure_paths={
            stage: [
                relative_artifact_path(path, root=artifacts.output_dir)
                for path in paths
            ]
            for stage, paths in figure_paths.items()
        },
    )
    write_json(artifacts.manifest_path, manifest.to_dict())

    return TransformWorkflowResult(
        output_path=artifacts.output_array_path,
        trace_path=artifacts.trace_path,
        manifest_path=artifacts.manifest_path,
        stage_array_paths=stage_array_paths,
        figure_paths=figure_paths,
        trace=traced.trace,
    )


def _run_traced_transform(
    pft: PyPFT,
    values: Any,
    *,
    direction: str,
) -> TracedTransformResult:
    if direction == "forward":
        return pft.forward_with_trace(values)
    return pft.backward_with_trace(values)


def _save_stage_arrays(
    trace: TransformTrace,
    output_dir: Path,
    *,
    enabled: bool,
) -> dict[str, Path]:
    if not enabled:
        return {}

    saved: dict[str, Path] = {}
    for index, frame in enumerate(trace.frames, start=1):
        path = build_stage_array_path(
            output_dir,
            index=index,
            stage=frame.stage,
        )
        save_array(path, squeeze_single_sample(frame.asarray(copy=False)))
        saved[frame.stage] = path
    return saved


def _serialize_trace(
    trace: TransformTrace,
    *,
    artifact_root: Path,
    stage_array_paths: dict[str, Path],
    figure_paths: dict[str, tuple[Path, ...]],
) -> dict[str, object]:
    frames = []
    for index, frame in enumerate(trace.frames, start=1):
        array = squeeze_single_sample(frame.asarray(copy=False))
        frames.append(
            {
                "index": index,
                "stage": frame.stage,
                "direction": frame.direction,
                "field_kind": frame.field_kind,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "grid": asdict(frame.grid),
                "array_path": _optional_path(
                    stage_array_paths.get(frame.stage),
                    root=artifact_root,
                ),
                "figure_paths": [
                    relative_artifact_path(path, root=artifact_root)
                    for path in figure_paths.get(frame.stage, ())
                ],
            }
        )

    return {
        "schema_version": 2,
        "artifact_root": ".",
        "direction": trace.direction,
        "frames": frames,
    }


def _output_array(values: Any):
    return values


def _optional_path(path: Path | None, *, root: Path) -> str | None:
    if path is None:
        return None
    return relative_artifact_path(path, root=root)


__all__ = ["run_transform_workflow"]
