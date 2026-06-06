from __future__ import annotations

import json
from pathlib import Path

import pytest

from pypft.core.exceptions import MetadataValidationError
from pypft.io import load_transform_metadata


def test_load_transform_metadata_accepts_matching_sidecar(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "sample.npy"
    input_path.write_bytes(b"placeholder")
    metadata_path = input_path.with_suffix(".pypft.json")
    metadata_path.write_text(
        json.dumps(
            {
                "domain": "spatial",
                "spatial_grid": {"radial_size": 3, "angular_size": 4},
                "frequency_grid": {"radial_size": 3, "angular_size": 4},
            }
        ),
        encoding="utf-8",
    )

    metadata, resolved_path = load_transform_metadata(
        input_path,
        expected_domain="spatial",
        runtime_shape=(3, 4),
    )

    assert metadata.domain == "spatial"
    assert metadata.spatial_grid.shape == (3, 4)
    assert resolved_path == metadata_path


def test_load_transform_metadata_rejects_grid_shape_mismatch(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "sample.npy"
    input_path.write_bytes(b"placeholder")
    input_path.with_suffix(".pypft.json").write_text(
        json.dumps(
            {
                "domain": "spatial",
                "spatial_grid": {"radial_size": 8, "angular_size": 4},
                "frequency_grid": {"radial_size": 8, "angular_size": 4},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MetadataValidationError):
        load_transform_metadata(
            input_path,
            expected_domain="spatial",
            runtime_shape=(3, 4),
        )
