from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TransformArtifactLayout:
    output_dir: Path
    output_array_path: Path
    trace_path: Path
    manifest_path: Path
    stage_array_dir: Path
    figure_dir: Path


def prepare_transform_artifacts(output_dir: Path) -> TransformArtifactLayout:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_array_dir = output_dir / "stages"
    figure_dir = output_dir / "figures"
    stage_array_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    return TransformArtifactLayout(
        output_dir=output_dir,
        output_array_path=output_dir / "output.npy",
        trace_path=output_dir / "trace.json",
        manifest_path=output_dir / "manifest.json",
        stage_array_dir=stage_array_dir,
        figure_dir=figure_dir,
    )


__all__ = ["TransformArtifactLayout", "prepare_transform_artifacts"]
