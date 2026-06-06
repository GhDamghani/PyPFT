from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pypft.cli.app import app, visualize_trace


def test_validate_roundtrip_cli_writes_report_and_figures(
    tmp_path: Path,
    roundtrip_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, roundtrip_image)
    radial_size, angular_size = roundtrip_image.shape
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

    runner = CliRunner()
    output_dir = tmp_path / "validation"
    result = runner.invoke(
        app,
        [
            "validate",
            "roundtrip",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "roundtrip-report.json").exists()
    assert len(list(output_dir.glob("*.png"))) == 2
    assert "Status" in result.output
    assert "pass" in result.output


def test_validate_compare_cli_writes_real_comparison_plot(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, np.array([[0.0, 1.0], [2.0, 3.0]]))
    np.save(candidate_path, np.array([[0.0, 1.5], [2.0, 3.0]]))

    runner = CliRunner()
    output_dir = tmp_path / "comparison"
    result = runner.invoke(
        app,
        [
            "validate",
            "compare",
            str(reference_path),
            str(candidate_path),
            "--output-dir",
            str(output_dir),
            "--atol",
            "0.1",
            "--rtol",
            "0.0",
        ],
    )

    assert result.exit_code == 1, result.output
    assert (output_dir / "comparison-report.json").exists()
    assert len(list(output_dir.glob("*.png"))) == 1
    assert "fail" in result.output


def test_validate_roundtrip_cli_help_documents_manifest_fallbacks() -> None:
    signature = inspect.signature(visualize_trace)
    gamma_option = signature.parameters["gamma"].default
    complex_view_option = signature.parameters["complex_view"].default

    assert "Override the gamma stored in the trace manifest" in (
        gamma_option.help
    )
    assert "Defaults to the manifest value" in gamma_option.help
    assert "Override the complex view stored in the trace manifest" in (
        complex_view_option.help
    )
    assert "Defaults to the manifest value" in complex_view_option.help
