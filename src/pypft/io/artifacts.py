from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


def _sanitize_token(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    return sanitized.strip("-.") or "artifact"


def _stage_stem(index: int, stage: str) -> str:
    return f"{index:02d}_{_sanitize_token(stage)}"


def build_field_figure_path(
    output_dir: Path,
    *,
    stem: str,
    view: str,
) -> Path:
    return output_dir / f"{_sanitize_token(stem)}.{_sanitize_token(view)}.png"


def build_trace_figure_path(
    figure_dir: Path,
    *,
    index: int,
    stage: str,
    view: str,
) -> Path:
    return figure_dir / (
        f"{_stage_stem(index, stage)}.{_sanitize_token(view)}.png"
    )


def build_stage_array_path(
    stage_array_dir: Path,
    *,
    index: int,
    stage: str,
) -> Path:
    return stage_array_dir / f"{_stage_stem(index, stage)}.npy"


def relative_artifact_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def resolve_artifact_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


@dataclass(frozen=True, slots=True)
class TransformArtifactLayout:
    output_dir: Path
    output_array_path: Path
    trace_path: Path
    manifest_path: Path
    stage_array_dir: Path
    figure_dir: Path

    def stage_array_path(self, *, index: int, stage: str) -> Path:
        return build_stage_array_path(
            self.stage_array_dir,
            index=index,
            stage=stage,
        )

    def trace_figure_path(
        self,
        *,
        index: int,
        stage: str,
        view: str,
    ) -> Path:
        return build_trace_figure_path(
            self.figure_dir,
            index=index,
            stage=stage,
            view=view,
        )


def prepare_transform_artifacts(output_dir: Path) -> TransformArtifactLayout:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_array_dir = output_dir / "stages"
    figure_dir = output_dir / "figures"
    return TransformArtifactLayout(
        output_dir=output_dir,
        output_array_path=output_dir / "output.npy",
        trace_path=output_dir / "trace.json",
        manifest_path=output_dir / "manifest.json",
        stage_array_dir=stage_array_dir,
        figure_dir=figure_dir,
    )


__all__ = [
    "TransformArtifactLayout",
    "build_field_figure_path",
    "build_stage_array_path",
    "build_trace_figure_path",
    "prepare_transform_artifacts",
    "relative_artifact_path",
    "resolve_artifact_path",
]
