from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pypft.cli.app import app


def test_transform_forward_cli_requires_output_dir(tmp_path: Path) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.ones((3, 4), dtype=np.complex128))
    _write_metadata(input_path.with_suffix(".pypft.json"), domain="spatial")

    result = CliRunner().invoke(app, ["transform", "forward", str(input_path)])

    assert result.exit_code == 2


def test_transform_forward_cli_rejects_missing_input_file(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "transform",
            "forward",
            str(tmp_path / "missing.npy"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2


def test_visualize_field_cli_accepts_angular_alias(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "field.npy"
    np.save(input_path, sample_image)

    output_dir = tmp_path / "figures"
    result = CliRunner().invoke(
        app,
        [
            "visualize",
            "field",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--complex-view",
            "angular",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "field.field.phase.png").exists()
    assert len(list(output_dir.glob("*.png"))) == 1


def test_inspect_trace_cli_rejects_missing_source(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["inspect", "trace", str(tmp_path / "missing-manifest.json")],
    )

    assert result.exit_code == 2


def test_validate_roundtrip_cli_requires_output_dir(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "input.npy"
    np.save(input_path, sample_image)
    _write_metadata(input_path.with_suffix(".pypft.json"), domain="spatial")

    result = CliRunner().invoke(
        app,
        ["validate", "roundtrip", str(input_path)],
    )

    assert result.exit_code == 2


def _write_metadata(path: Path, *, domain: str) -> None:
    path.write_text(
        json.dumps(
            {
                "domain": domain,
                "spatial_grid": {"radial_size": 3, "angular_size": 4},
                "frequency_grid": {"radial_size": 3, "angular_size": 4},
            }
        ),
        encoding="utf-8",
    )
