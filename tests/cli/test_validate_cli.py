from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pypft.cli.app import app


def test_validate_roundtrip_cli_writes_report_and_figures(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, sample_image)
    input_path.with_suffix(".pypft.json").write_text(
        json.dumps(
            {
                "domain": "spatial",
                "spatial_grid": {"radial_size": 3, "angular_size": 4},
                "frequency_grid": {"radial_size": 3, "angular_size": 4},
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

    assert result.exit_code == 0, result.output
    assert (output_dir / "comparison-report.json").exists()
    assert len(list(output_dir.glob("*.png"))) == 1
    assert "fail" in result.output
