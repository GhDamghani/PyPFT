from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pypft.io import load_transform_run_manifest
from pypft.core.exceptions import MetadataValidationError
from pypft.tracing import load_saved_trace
from pypft.visualization import FieldRenderSpec, save_trace_figures
from pypft.workflows import TransformWorkflowRequest, run_transform_workflow


def test_saved_trace_can_be_loaded_and_rendered(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, sample_image)
    radial_size, angular_size = sample_image.shape
    input_path.with_suffix(".pypft.json").write_text(
        json.dumps(
            {
                "domain": "spatial",
                "spatial_grid": {
                    "radial_size": radial_size,
                    "angular_size": angular_size,
                },
                "frequency_grid": {
                    "radial_size": radial_size,
                    "angular_size": angular_size,
                },
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
            save_all_views=False,
            save_stage_arrays=True,
        )
    )

    manifest = load_transform_run_manifest(result.manifest_path)
    replay_trace = load_saved_trace(result.trace_path)
    rendered = save_trace_figures(
        replay_trace,
        tmp_path / "replayed-figures",
        render_spec=FieldRenderSpec(complex_view="both", gamma=0.5),
        save_all_views=False,
    )

    assert manifest.direction == "forward"
    assert manifest.stage_array_paths["spatial_samples"].endswith(
        "01_spatial_samples.npy"
    )
    assert replay_trace.stage_names == (
        "spatial_samples",
        "angular_dft",
        "radial_dht",
        "angular_idft",
        "frequency_samples",
    )
    assert len(rendered["frequency_samples"]) == 2
    for path in rendered["frequency_samples"]:
        assert path.exists()


def test_saved_trace_requires_stage_arrays_for_replay(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, sample_image)
    radial_size, angular_size = sample_image.shape
    input_path.with_suffix(".pypft.json").write_text(
        json.dumps(
            {
                "domain": "spatial",
                "spatial_grid": {
                    "radial_size": radial_size,
                    "angular_size": angular_size,
                },
                "frequency_grid": {
                    "radial_size": radial_size,
                    "angular_size": angular_size,
                },
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
            save_all_views=False,
            save_stage_arrays=False,
        )
    )

    with pytest.raises(
        MetadataValidationError,
        match="Re-run with --save-stage-arrays to enable trace replay",
    ):
        load_saved_trace(result.trace_path)
