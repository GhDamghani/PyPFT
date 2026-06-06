from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pypft.fields import (
    AngularSpectrum,
    FrequencySamples,
    RadialSpectrum,
    SpatialSamples,
)

TraceStage: TypeAlias = Literal[
    "spatial_samples",
    "angular_dft",
    "radial_dht",
    "angular_idft",
    "frequency_samples",
]
TransformDirection: TypeAlias = Literal["forward", "backward"]
TracedField: TypeAlias = (
    SpatialSamples | AngularSpectrum | RadialSpectrum | FrequencySamples
)


def _field_kind(field: TracedField) -> str:
    if isinstance(field, SpatialSamples):
        return "spatial_samples"
    if isinstance(field, AngularSpectrum):
        return "angular_spectrum"
    if isinstance(field, RadialSpectrum):
        return "radial_spectrum"
    return "frequency_samples"


@dataclass(frozen=True, slots=True)
class TraceFrame:
    stage: TraceStage
    direction: TransformDirection
    field: TracedField

    @property
    def field_kind(self) -> str:
        return _field_kind(self.field)

    @property
    def grid(self) -> object:
        return self.field.grid

    def asarray(self, *, copy: bool = False) -> Any:
        return self.field.asarray(copy=copy)


@dataclass(frozen=True, slots=True)
class TransformTrace:
    direction: TransformDirection
    frames: tuple[TraceFrame, ...]

    @property
    def stage_names(self) -> tuple[TraceStage, ...]:
        return tuple(frame.stage for frame in self.frames)

    @property
    def final_frame(self) -> TraceFrame:
        return self.frames[-1]


@dataclass(frozen=True, slots=True)
class TracedTransformResult:
    output: Any
    trace: TransformTrace
    had_batch_axis: bool


__all__ = [
    "TraceFrame",
    "TraceStage",
    "TracedField",
    "TracedTransformResult",
    "TransformDirection",
    "TransformTrace",
]
