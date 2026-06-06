from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pypft.workflows import TransformWorkflowRequest, run_transform_workflow


def test_transform_workflow_request_normalizes_angular_alias() -> None:
    request = TransformWorkflowRequest(
        direction="forward",
        input_path=Path("input.npy"),
        output_dir=Path("artifacts"),
        complex_view="angular",
    )

    assert request.complex_view == "phase"


def test_run_transform_workflow_saves_phase_one_artifacts(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, sample_image)
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

    result = run_transform_workflow(
        TransformWorkflowRequest(
            direction="forward",
            input_path=input_path,
            output_dir=tmp_path / "artifacts",
            gamma=0.5,
            complex_view="both",
            save_all_views=True,
            save_stage_arrays=True,
        )
    )

    assert result.output_path.exists()
    assert result.trace_path.exists()
    assert result.manifest_path.exists()
    assert set(result.stage_array_paths) == {
        "spatial_samples",
        "angular_dft",
        "radial_dht",
        "angular_idft",
        "frequency_samples",
    }
    assert set(result.figure_paths) == {
        "spatial_samples",
        "angular_dft",
        "radial_dht",
        "angular_idft",
        "frequency_samples",
    }

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["artifact_root"] == "."
    assert manifest["output_array_path"] == "output.npy"
    assert manifest["trace_path"] == "trace.json"
    assert manifest["trace_stage_names"] == [
        "spatial_samples",
        "angular_dft",
        "radial_dht",
        "angular_idft",
        "frequency_samples",
    ]
    assert manifest["stage_array_paths"]["spatial_samples"] == (
        "stages/01_spatial_samples.npy"
    )
    assert manifest["figure_paths"]["spatial_samples"] == [
        "figures/01_spatial_samples.magnitude.png",
        "figures/01_spatial_samples.phase.png",
    ]
    assert trace["schema_version"] == 2
    assert trace["artifact_root"] == "."
    assert trace["frames"][0]["array_path"] == "stages/01_spatial_samples.npy"
    assert trace["frames"][0]["figure_paths"] == [
        "figures/01_spatial_samples.magnitude.png",
        "figures/01_spatial_samples.phase.png",
    ]
