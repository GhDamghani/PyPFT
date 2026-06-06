from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pypft import PyPFT
from pypft.cli.app import app


def test_transform_forward_cli_saves_all_requested_artifacts(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "spatial.npy"
    np.save(input_path, sample_image)
    _write_metadata(
        input_path.with_suffix(".pypft.json"),
        domain="spatial",
        shape=sample_image.shape,
    )

    runner = CliRunner()
    output_dir = tmp_path / "forward-output"
    result = runner.invoke(
        app,
        [
            "transform",
            "forward",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--gamma",
            "0.5",
            "--save-all-views",
            "--save-stage-arrays",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "output.npy").exists()
    assert (output_dir / "trace.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert len(list((output_dir / "stages").glob("*.npy"))) == 5
    assert len(list((output_dir / "figures").glob("*.png"))) == 10


def test_transform_backward_cli_accepts_angular_alias(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    input_path = tmp_path / "frequency.npy"
    np.save(input_path, PyPFT().forward(sample_image))
    _write_metadata(
        input_path.with_suffix(".pypft.json"),
        domain="frequency",
        shape=sample_image.shape,
    )

    runner = CliRunner()
    output_dir = tmp_path / "backward-output"
    result = runner.invoke(
        app,
        [
            "transform",
            "backward",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--complex-view",
            "angular",
            "--save-all-views",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list((output_dir / "figures").glob("*.png"))) == 5


def test_transform_cli_requires_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "spatial.npy"
    np.save(input_path, np.ones((3, 4), dtype=np.complex128))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "transform",
            "forward",
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "sidecar metadata" in result.output.lower()


def _write_metadata(
    path: Path,
    *,
    domain: str,
    shape: tuple[int, int],
) -> None:
    radial_size, angular_size = shape
    path.write_text(
        json.dumps(
            {
                "domain": domain,
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
