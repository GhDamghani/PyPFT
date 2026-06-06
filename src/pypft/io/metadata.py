from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

from pypft.core.exceptions import MetadataValidationError
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid

MetadataDomain = Literal["spatial", "frequency"]


@dataclass(frozen=True, slots=True)
class TransformMetadata:
    schema_version: int
    domain: MetadataDomain
    spatial_grid: PolarSpatialGrid
    frequency_grid: PolarFrequencyGrid
    backend: str | None = None
    dht_implementation: str | None = None
    conventions: dict[str, object] | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_metadata_path(
    input_path: Path,
    metadata_path: Path | None = None,
) -> Path:
    if metadata_path is not None:
        return metadata_path
    return input_path.with_suffix(".pypft.json")


def load_transform_metadata(
    input_path: Path,
    *,
    expected_domain: MetadataDomain,
    runtime_shape: tuple[int, int],
    metadata_path: Path | None = None,
) -> tuple[TransformMetadata, Path]:
    resolved_path = resolve_metadata_path(input_path, metadata_path)
    if not resolved_path.exists():
        raise MetadataValidationError(
            "PyPFT CLI requires sidecar metadata for scientific runs. "
            f"Expected {resolved_path}."
        )

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MetadataValidationError(
            f"Metadata file {resolved_path} is not valid JSON."
        ) from error

    metadata = _parse_metadata(payload)
    if metadata.domain != expected_domain:
        raise MetadataValidationError(
            f"Metadata domain {metadata.domain!r} does not match the "
            f"requested {expected_domain!r} transform command."
        )

    expected_shape = (
        metadata.spatial_grid.shape
        if expected_domain == "spatial"
        else metadata.frequency_grid.shape
    )
    if expected_shape != runtime_shape:
        raise MetadataValidationError(
            f"Metadata grid shape {expected_shape!r} does not match the "
            f"runtime .npy shape {runtime_shape!r}."
        )

    return metadata, resolved_path


def _parse_metadata(payload: object) -> TransformMetadata:
    if not isinstance(payload, dict):
        raise MetadataValidationError("Metadata JSON must be an object.")

    schema_version = payload.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version != 1:
        raise MetadataValidationError(
            "Metadata schema_version must be the integer 1."
        )

    domain = payload.get("domain")
    if domain not in {"spatial", "frequency"}:
        raise MetadataValidationError(
            "Metadata must declare domain as 'spatial' or 'frequency'."
        )

    backend = _optional_string(payload.get("backend"), field_name="backend")
    dht_implementation = _optional_string(
        payload.get("dht_implementation"),
        field_name="dht_implementation",
    )
    conventions = payload.get("conventions")
    if conventions is not None and not isinstance(conventions, dict):
        raise MetadataValidationError(
            "Metadata conventions must be an object when provided."
        )

    notes_payload = payload.get("notes", [])
    if not isinstance(notes_payload, list) or not all(
        isinstance(note, str) for note in notes_payload
    ):
        raise MetadataValidationError(
            "Metadata notes must be a list of strings when provided."
        )

    return TransformMetadata(
        schema_version=schema_version,
        domain=domain,
        spatial_grid=_parse_spatial_grid(payload.get("spatial_grid")),
        frequency_grid=_parse_frequency_grid(payload.get("frequency_grid")),
        backend=backend,
        dht_implementation=dht_implementation,
        conventions=conventions,
        notes=tuple(notes_payload),
    )


def _parse_spatial_grid(payload: object) -> PolarSpatialGrid:
    radial_size, angular_size = _parse_grid_sizes(
        payload,
        field_name="spatial_grid",
    )
    return PolarSpatialGrid(radial_size=radial_size, angular_size=angular_size)


def _parse_frequency_grid(payload: object) -> PolarFrequencyGrid:
    radial_size, angular_size = _parse_grid_sizes(
        payload,
        field_name="frequency_grid",
    )
    return PolarFrequencyGrid(
        radial_size=radial_size,
        angular_size=angular_size,
    )


def _parse_grid_sizes(
    payload: object,
    *,
    field_name: str,
) -> tuple[int, int]:
    if not isinstance(payload, dict):
        raise MetadataValidationError(
            f"Metadata {field_name} must be an object with radial_size and "
            "angular_size."
        )

    radial_size = payload.get("radial_size")
    angular_size = payload.get("angular_size")
    if not isinstance(radial_size, int) or not isinstance(angular_size, int):
        raise MetadataValidationError(
            f"Metadata {field_name} must use integer radial_size and "
            "angular_size values."
        )
    return radial_size, angular_size


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MetadataValidationError(
            f"Metadata {field_name} must be a string when provided."
        )
    return value


__all__ = [
    "MetadataDomain",
    "TransformMetadata",
    "load_transform_metadata",
    "resolve_metadata_path",
]
