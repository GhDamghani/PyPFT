from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from pypft.core.exceptions import MetadataValidationError
from pypft.io.artifacts import resolve_artifact_path
from pypft.tracing.models import TransformDirection

ReplayFrameDirection: TypeAlias = TransformDirection | Literal["unknown"]


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    stage: str
    direction: ReplayFrameDirection
    field_kind: str
    values: np.ndarray
    grid: dict[str, object]

    def asarray(self, *, copy: bool = False) -> np.ndarray:
        if copy:
            return np.array(self.values, copy=True)
        return self.values


@dataclass(frozen=True, slots=True)
class ReplayTrace:
    direction: TransformDirection
    frames: tuple[ReplayFrame, ...]

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(frame.stage for frame in self.frames)

    @property
    def final_frame(self) -> ReplayFrame:
        return self.frames[-1]


@dataclass(frozen=True, slots=True)
class StoredTraceFrame:
    index: int
    stage: str
    direction: TransformDirection
    field_kind: str
    shape: tuple[int, ...]
    dtype: str
    grid: dict[str, object]
    array_path: str | None
    figure_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StoredTraceDocument:
    schema_version: int
    direction: TransformDirection
    artifact_root: str
    frames: tuple[StoredTraceFrame, ...]


def load_stored_trace_document(path: Path) -> StoredTraceDocument:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Trace file {path} is not valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise MetadataValidationError("Trace JSON must be an object.")

    direction = payload.get("direction")
    if direction not in {"forward", "backward"}:
        raise MetadataValidationError(
            "Trace direction must be 'forward' or 'backward'."
        )

    artifact_root = payload.get("artifact_root", ".")
    if not isinstance(artifact_root, str):
        raise MetadataValidationError("Trace artifact_root must be a string.")

    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list):
        raise MetadataValidationError("Trace frames must be a list.")

    frames: list[StoredTraceFrame] = []
    for frame_payload in frames_payload:
        if not isinstance(frame_payload, dict):
            raise MetadataValidationError(
                "Trace frames must be JSON objects."
            )

        index = frame_payload.get("index")
        stage = frame_payload.get("stage")
        frame_direction = frame_payload.get("direction")
        field_kind = frame_payload.get("field_kind")
        shape = frame_payload.get("shape")
        dtype = frame_payload.get("dtype")
        grid = frame_payload.get("grid")
        array_path = frame_payload.get("array_path")
        figure_paths = frame_payload.get("figure_paths", ())
        if not isinstance(index, int):
            raise MetadataValidationError("Trace frame index must be an int.")
        if not isinstance(stage, str):
            raise MetadataValidationError(
                "Trace frame stage must be a string."
            )
        if frame_direction not in {"forward", "backward"}:
            raise MetadataValidationError(
                "Trace frame direction must be 'forward' or 'backward'."
            )
        if not isinstance(field_kind, str):
            raise MetadataValidationError(
                "Trace frame field_kind must be a string."
            )
        if not isinstance(shape, list) or not all(
            isinstance(item, int) for item in shape
        ):
            raise MetadataValidationError(
                "Trace frame shape must be a list of ints."
            )
        if not isinstance(dtype, str):
            raise MetadataValidationError(
                "Trace frame dtype must be a string."
            )
        if not isinstance(grid, dict):
            raise MetadataValidationError(
                "Trace frame grid must be an object."
            )
        if array_path is not None and not isinstance(array_path, str):
            raise MetadataValidationError(
                "Trace frame array_path must be a string or null."
            )
        if not isinstance(figure_paths, list) or not all(
            isinstance(item, str) for item in figure_paths
        ):
            raise MetadataValidationError(
                "Trace frame figure_paths must be a list of strings."
            )

        frames.append(
            StoredTraceFrame(
                index=index,
                stage=stage,
                direction=frame_direction,
                field_kind=field_kind,
                shape=tuple(shape),
                dtype=dtype,
                grid=grid,
                array_path=array_path,
                figure_paths=tuple(figure_paths),
            )
        )

    return StoredTraceDocument(
        schema_version=int(payload.get("schema_version", 1)),
        direction=direction,
        artifact_root=artifact_root,
        frames=tuple(frames),
    )


def load_saved_trace(path: Path) -> ReplayTrace:
    document = load_stored_trace_document(path)
    artifact_root = resolve_artifact_path(
        document.artifact_root,
        root=path.parent,
    )
    frames: list[ReplayFrame] = []
    for frame in document.frames:
        if frame.array_path is None:
            raise MetadataValidationError(
                "Saved trace replay requires frame array_path strings."
            )

        resolved_array_path = resolve_artifact_path(
            frame.array_path,
            root=artifact_root,
        )
        values = np.load(resolved_array_path, allow_pickle=False)
        frames.append(
            ReplayFrame(
                stage=frame.stage,
                direction=frame.direction,
                field_kind=frame.field_kind,
                values=np.asarray(values),
                grid=frame.grid,
            )
        )

    return ReplayTrace(direction=document.direction, frames=tuple(frames))


__all__ = [
    "ReplayFrame",
    "ReplayTrace",
    "StoredTraceDocument",
    "StoredTraceFrame",
    "load_saved_trace",
    "load_stored_trace_document",
]
