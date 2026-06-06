from __future__ import annotations

from dataclasses import dataclass, field

from pypft.fields import FrequencySamples, SpatialSamples
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.operators.angular import AngularDFT, AngularIDFT
from pypft.operators.radial import RadialDHT
from pypft.tracing import TraceFrame, TraceSink


@dataclass(frozen=True, slots=True)
class ForwardPFTPlan:
    angular_transform: AngularDFT = field(default_factory=AngularDFT)
    radial_transform: RadialDHT = field(default_factory=RadialDHT)
    angular_reconstruction: AngularIDFT = field(default_factory=AngularIDFT)

    def execute(
        self,
        values: SpatialSamples,
        *,
        frequency_grid: PolarFrequencyGrid,
        trace_sink: TraceSink | None = None,
    ) -> FrequencySamples:
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="spatial_samples",
                direction="forward",
                field=values,
            ),
        )
        angular_modes = self.angular_transform.apply(values)
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="angular_dft",
                direction="forward",
                field=angular_modes,
            ),
        )
        radial_modes = self.radial_transform.apply(
            angular_modes,
            target_grid=frequency_grid,
        )
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="radial_dht",
                direction="forward",
                field=radial_modes,
            ),
        )
        reconstructed = self.angular_reconstruction.apply(radial_modes)
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="angular_idft",
                direction="forward",
                field=reconstructed,
            ),
        )
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="frequency_samples",
                direction="forward",
                field=reconstructed,
            ),
        )
        return reconstructed


@dataclass(frozen=True, slots=True)
class BackwardPFTPlan:
    angular_transform: AngularDFT = field(default_factory=AngularDFT)
    radial_transform: RadialDHT = field(default_factory=RadialDHT)
    angular_reconstruction: AngularIDFT = field(default_factory=AngularIDFT)

    def execute(
        self,
        values: FrequencySamples,
        *,
        spatial_grid: PolarSpatialGrid,
        trace_sink: TraceSink | None = None,
    ) -> SpatialSamples:
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="frequency_samples",
                direction="backward",
                field=values,
            ),
        )
        radial_modes = self.angular_transform.apply(values)
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="angular_dft",
                direction="backward",
                field=radial_modes,
            ),
        )
        angular_modes = self.radial_transform.apply(
            radial_modes,
            target_grid=spatial_grid,
        )
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="radial_dht",
                direction="backward",
                field=angular_modes,
            ),
        )
        reconstructed = self.angular_reconstruction.apply(angular_modes)
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="angular_idft",
                direction="backward",
                field=reconstructed,
            ),
        )
        _record_trace(
            trace_sink,
            TraceFrame(
                stage="spatial_samples",
                direction="backward",
                field=reconstructed,
            ),
        )
        return reconstructed


def _record_trace(
    trace_sink: TraceSink | None,
    frame: TraceFrame,
) -> None:
    if trace_sink is None:
        return
    trace_sink.record(frame)


__all__ = ["BackwardPFTPlan", "ForwardPFTPlan"]
