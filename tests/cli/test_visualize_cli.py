from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pypft.cli.app import app
from pypft.workflows import TransformWorkflowRequest, run_transform_workflow


def test_visualize_field_cli_writes_expected_figures(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "field.npy"
    np.save(input_path, sample_image)

    runner = CliRunner()
    output_dir = tmp_path / "field-figures"
    result = runner.invoke(
        app,
        [
            "visualize",
            "field",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--field-kind",
            "spatial_samples",
            "--complex-view",
            "both",
            "--gamma",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(output_dir.glob("*.png"))) == 2


def test_visualize_trace_cli_replays_from_manifest_without_stage_arrays(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    manifest_path = _write_transform_artifacts(
        tmp_path,
        sample_image,
        save_stage_arrays=False,
    )

    runner = CliRunner()
    output_dir = tmp_path / "replayed-figures"
    result = runner.invoke(
        app,
        [
            "visualize",
            "trace",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(output_dir.glob("*.png"))) == 10


def test_visualize_trace_cli_renders_saved_trace_document(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    manifest_path = _write_transform_artifacts(
        tmp_path,
        sample_image,
        save_stage_arrays=True,
    )
    trace_path = manifest_path.parent / "trace.json"

    runner = CliRunner()
    output_dir = tmp_path / "trace-figures"
    result = runner.invoke(
        app,
        [
            "visualize",
            "trace",
            str(trace_path),
            "--output-dir",
            str(output_dir),
            "--complex-view",
            "phase",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(output_dir.glob("*.png"))) == 5


def test_inspect_trace_cli_shows_canonical_stage_names(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    manifest_path = _write_transform_artifacts(
        tmp_path,
        sample_image,
        save_stage_arrays=False,
    )

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "trace", str(manifest_path)])

    assert result.exit_code == 0, result.output
    assert "spatial_samples" in result.output
    assert "angular_dft" in result.output
    assert "frequency_samples" in result.output


def _write_transform_artifacts(
    tmp_path: Path,
    sample_image: np.ndarray,
    *,
    save_stage_arrays: bool,
) -> Path:
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
            save_stage_arrays=save_stage_arrays,
        )
    )
    return result.manifest_path
