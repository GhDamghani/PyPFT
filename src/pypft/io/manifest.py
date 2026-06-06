from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from pypft.io.metadata import TransformMetadata
from pypft.tracing import TransformDirection
from pypft.visualization.models import ComplexView


@dataclass(frozen=True, slots=True)
class TransformRunManifest:
    schema_version: int
    direction: TransformDirection
    input_path: str
    metadata_path: str
    output_array_path: str
    trace_path: str
    manifest_path: str
    package_version: str
    backend: str
    dht_implementation: str
    gamma: float
    complex_view: ComplexView
    save_all_views: bool
    save_stage_arrays: bool
    metadata: TransformMetadata
    stage_array_paths: dict[str, str]
    figure_paths: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["TransformRunManifest", "write_json"]
