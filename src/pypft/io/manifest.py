from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from pypft.core.exceptions import MetadataValidationError
from pypft.io.artifacts import resolve_artifact_path
from pypft.io.metadata import TransformMetadata, parse_transform_metadata
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
    artifact_root: str = "."
    trace_stage_names: tuple[str, ...] = ()
    stage_array_paths: dict[str, str] = field(default_factory=dict)
    figure_paths: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload

    def resolve_input_path(self, *, manifest_path: Path) -> Path:
        return resolve_artifact_path(
            self.input_path,
            root=manifest_path.parent,
        )

    def resolve_metadata_path(self, *, manifest_path: Path) -> Path:
        return resolve_artifact_path(
            self.metadata_path,
            root=manifest_path.parent,
        )

    def resolve_artifact_root(self, *, manifest_path: Path) -> Path:
        return resolve_artifact_path(
            self.artifact_root,
            root=manifest_path.parent,
        )

    def resolve_trace_path(self, *, manifest_path: Path) -> Path:
        return resolve_artifact_path(
            self.trace_path,
            root=self.resolve_artifact_root(manifest_path=manifest_path),
        )


def load_transform_run_manifest(path: Path) -> TransformRunManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Manifest file {path} is not valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise MetadataValidationError("Manifest JSON must be an object.")

    try:
        metadata_payload = payload["metadata"]
        direction = payload["direction"]
        input_path = payload["input_path"]
        metadata_path = payload["metadata_path"]
        output_array_path = payload["output_array_path"]
        trace_path = payload["trace_path"]
        manifest_path = payload["manifest_path"]
        package_version = payload["package_version"]
        backend = payload["backend"]
        dht_implementation = payload["dht_implementation"]
        gamma = payload["gamma"]
        complex_view = payload["complex_view"]
        save_all_views = payload["save_all_views"]
        save_stage_arrays = payload["save_stage_arrays"]
        artifact_root = payload.get("artifact_root", ".")
        trace_stage_names = payload.get("trace_stage_names", ())
        stage_array_paths = payload.get("stage_array_paths", {})
        figure_paths = payload.get("figure_paths", {})
    except KeyError as error:
        raise MetadataValidationError(
            f"Manifest file {path} is missing required key {error.args[0]!r}."
        ) from error

    if direction not in {"forward", "backward"}:
        raise MetadataValidationError(
            "Manifest direction must be 'forward' or 'backward'."
        )
    if not isinstance(input_path, str) or not isinstance(metadata_path, str):
        raise MetadataValidationError(
            "Manifest input_path and metadata_path must be strings."
        )
    if not isinstance(output_array_path, str) or not isinstance(
        trace_path,
        str,
    ):
        raise MetadataValidationError(
            "Manifest artifact paths must be strings."
        )
    if not isinstance(manifest_path, str) or not isinstance(
        package_version,
        str,
    ):
        raise MetadataValidationError(
            "Manifest manifest_path and package_version must be strings."
        )
    if not isinstance(backend, str) or not isinstance(dht_implementation, str):
        raise MetadataValidationError(
            "Manifest backend and dht_implementation must be strings."
        )
    if not isinstance(artifact_root, str):
        raise MetadataValidationError(
            "Manifest artifact_root must be a string."
        )
    if not isinstance(gamma, (int, float)):
        raise MetadataValidationError("Manifest gamma must be numeric.")
    if complex_view not in {"magnitude", "phase", "both"}:
        raise MetadataValidationError(
            "Manifest complex_view must be one of 'magnitude', 'phase', "
            "or 'both'."
        )
    if not isinstance(save_all_views, bool) or not isinstance(
        save_stage_arrays,
        bool,
    ):
        raise MetadataValidationError(
            "Manifest save_all_views and save_stage_arrays must be booleans."
        )
    if not isinstance(trace_stage_names, (list, tuple)) or not all(
        isinstance(item, str) for item in trace_stage_names
    ):
        raise MetadataValidationError(
            "Manifest trace_stage_names must be a list or tuple of strings."
        )
    if not isinstance(stage_array_paths, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in stage_array_paths.items()
    ):
        raise MetadataValidationError(
            "Manifest stage_array_paths must be a mapping of strings to "
            "strings."
        )
    if not isinstance(figure_paths, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        for key, value in figure_paths.items()
    ):
        raise MetadataValidationError(
            "Manifest figure_paths must map strings to lists of strings."
        )

    return TransformRunManifest(
        schema_version=int(payload.get("schema_version", 1)),
        direction=direction,
        input_path=input_path,
        metadata_path=metadata_path,
        output_array_path=output_array_path,
        trace_path=trace_path,
        manifest_path=manifest_path,
        package_version=package_version,
        backend=backend,
        dht_implementation=dht_implementation,
        gamma=float(gamma),
        complex_view=complex_view,
        save_all_views=save_all_views,
        save_stage_arrays=save_stage_arrays,
        metadata=parse_transform_metadata(metadata_payload),
        artifact_root=artifact_root,
        trace_stage_names=tuple(trace_stage_names),
        stage_array_paths=stage_array_paths,
        figure_paths=figure_paths,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "TransformRunManifest",
    "load_transform_run_manifest",
    "write_json",
]
