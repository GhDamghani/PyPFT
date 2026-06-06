from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pypft.workflows import compare_field_files, validate_roundtrip


def test_validate_roundtrip_writes_report_and_comparison_figures(
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

    result = validate_roundtrip(
        input_path,
        tmp_path / "validation",
        gamma=0.5,
        complex_view="both",
        atol=1e-10,
        rtol=1e-10,
    )

    assert result.forward_path.exists()
    assert result.reconstruction_path.exists()
    assert result.report_path.exists()
    assert len(result.figure_paths) == 2
    assert result.metrics.passes_tolerance
    assert result.metrics.relative_l2_error < 1e-10

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["kind"] == "roundtrip_validation"
    assert report["forward_path"] == "forward.npy"
    assert report["reconstruction_path"] == "reconstruction.npy"
    assert report["figure_paths"] == [
        "roundtrip_comparison.magnitude.png",
        "roundtrip_comparison.phase.png",
    ]


def test_compare_field_files_reports_metrics_and_saves_real_plot(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    candidate = reference + 1.0
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)

    result = compare_field_files(
        reference_path,
        candidate_path,
        tmp_path / "comparison",
        field_kind="field",
        gamma=1.0,
        complex_view="both",
        atol=0.1,
        rtol=0.0,
    )

    assert result.report_path.exists()
    assert len(result.figure_paths) == 1
    assert not result.metrics.passes_tolerance
    assert result.metrics.max_abs_error == 1.0
    assert result.metrics.mean_abs_error == 1.0
    assert result.metrics.rmse == 1.0

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["kind"] == "field_comparison"
    assert report["figure_paths"] == ["comparison.real.png"]
