from __future__ import annotations

from pathlib import Path

import typer

from pypft.workflows import TransformWorkflowRequest


def build_transform_request(
    *,
    direction: str,
    input_path: Path,
    metadata_path: Path | None,
    output_dir: Path,
    gamma: float,
    complex_view: str,
    save_all_views: bool,
    save_stage_arrays: bool,
    backend: str | None,
    dht_implementation: str | None,
) -> TransformWorkflowRequest:
    return TransformWorkflowRequest(
        direction=direction,
        input_path=input_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        gamma=gamma,
        complex_view=complex_view,
        save_all_views=save_all_views,
        save_stage_arrays=save_stage_arrays,
        backend=backend,
        dht_implementation=dht_implementation,
    )


def validate_complex_view(value: str) -> str:
    from pypft.visualization import normalize_complex_view

    try:
        return normalize_complex_view(value)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


__all__ = ["build_transform_request", "validate_complex_view"]
