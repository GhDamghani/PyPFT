from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pypft import PyPFT
from pypft.core.exceptions import InputShapeError
from pypft.io import (
    load_array,
    load_transform_metadata,
    relative_artifact_path,
    save_array,
    write_json,
)
from pypft.tracing import ReplayFrame
from pypft.visualization import (
    FieldRenderSpec,
    normalize_complex_view,
    save_field_comparison,
    squeeze_single_sample,
)


@dataclass(frozen=True, slots=True)
class FieldComparisonMetrics:
    shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    relative_l2_error: float
    reference_l2_norm: float
    passes_tolerance: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class ComparisonWorkflowResult:
    report_path: Path
    figure_paths: tuple[Path, ...]
    metrics: FieldComparisonMetrics


@dataclass(frozen=True, slots=True)
class RoundtripValidationResult:
    forward_path: Path
    reconstruction_path: Path
    report_path: Path
    figure_paths: tuple[Path, ...]
    metrics: FieldComparisonMetrics


def compare_field_files(
    reference_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    field_kind: str,
    gamma: float,
    complex_view: str,
    atol: float,
    rtol: float,
) -> ComparisonWorkflowResult:
    reference = _load_2d_array(reference_path, label="reference")
    candidate = _load_2d_array(candidate_path, label="candidate")
    _validate_matching_shape(reference, candidate)

    metrics = _compute_metrics(reference, candidate, atol=atol, rtol=rtol)
    normalized_view = normalize_complex_view(complex_view)
    figure_paths = _save_comparison_figures(
        reference,
        candidate,
        field_kind=field_kind,
        output_dir=output_dir,
        gamma=gamma,
        complex_view=normalized_view,
        reference_stage="reference",
        candidate_stage="candidate",
        stem="comparison",
    )
    report_path = output_dir / "comparison-report.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "kind": "field_comparison",
            "reference_path": str(reference_path),
            "candidate_path": str(candidate_path),
            "field_kind": field_kind,
            "gamma": gamma,
            "complex_view": normalized_view,
            "figure_paths": [
                relative_artifact_path(path, root=output_dir)
                for path in figure_paths
            ],
            "metrics": metrics.to_dict(),
        },
    )
    return ComparisonWorkflowResult(
        report_path=report_path,
        figure_paths=figure_paths,
        metrics=metrics,
    )


def validate_roundtrip(
    input_path: Path,
    output_dir: Path,
    *,
    metadata_path: Path | None = None,
    gamma: float,
    complex_view: str,
    atol: float,
    rtol: float,
    backend: str | None = None,
    dht_implementation: str | None = None,
) -> RoundtripValidationResult:
    values = _load_2d_array(input_path, label="roundtrip")
    metadata, metadata_source = load_transform_metadata(
        input_path,
        expected_domain="spatial",
        runtime_shape=values.shape,
        metadata_path=metadata_path,
    )

    backend_name = backend or metadata.backend or "cpu"
    dht_name = dht_implementation or metadata.dht_implementation or "naive"
    normalized_view = normalize_complex_view(complex_view)

    pft = PyPFT(
        backend=backend_name,
        dht_implementation=dht_name,
        spatial_grid=metadata.spatial_grid,
        frequency_grid=metadata.frequency_grid,
    )
    forward = np.asarray(pft.forward(values))
    reconstruction = np.asarray(pft.backward(forward))

    forward_path = output_dir / "forward.npy"
    reconstruction_path = output_dir / "reconstruction.npy"
    save_array(forward_path, forward)
    save_array(reconstruction_path, reconstruction)

    metrics = _compute_metrics(values, reconstruction, atol=atol, rtol=rtol)
    figure_paths = _save_comparison_figures(
        values,
        reconstruction,
        field_kind="spatial_samples",
        output_dir=output_dir,
        gamma=gamma,
        complex_view=normalized_view,
        reference_stage="input",
        candidate_stage="reconstruction",
        stem="roundtrip_comparison",
    )
    report_path = output_dir / "roundtrip-report.json"
    write_json(
        report_path,
        {
            "schema_version": 1,
            "kind": "roundtrip_validation",
            "input_path": str(input_path),
            "metadata_path": str(metadata_source),
            "backend": backend_name,
            "dht_implementation": dht_name,
            "gamma": gamma,
            "complex_view": normalized_view,
            "forward_path": relative_artifact_path(
                forward_path,
                root=output_dir,
            ),
            "reconstruction_path": relative_artifact_path(
                reconstruction_path,
                root=output_dir,
            ),
            "figure_paths": [
                relative_artifact_path(path, root=output_dir)
                for path in figure_paths
            ],
            "metadata": metadata.to_dict(),
            "metrics": metrics.to_dict(),
        },
    )
    return RoundtripValidationResult(
        forward_path=forward_path,
        reconstruction_path=reconstruction_path,
        report_path=report_path,
        figure_paths=figure_paths,
        metrics=metrics,
    )


def _save_comparison_figures(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    field_kind: str,
    output_dir: Path,
    gamma: float,
    complex_view: str,
    reference_stage: str,
    candidate_stage: str,
    stem: str,
) -> tuple[Path, ...]:
    return save_field_comparison(
        ReplayFrame(
            stage=reference_stage,
            direction="forward",
            field_kind=field_kind,
            values=reference,
            grid={},
        ),
        ReplayFrame(
            stage=candidate_stage,
            direction="forward",
            field_kind=field_kind,
            values=candidate,
            grid={},
        ),
        output_dir,
        render_spec=FieldRenderSpec(
            complex_view=complex_view,
            gamma=gamma,
        ),
        stem=stem,
    )


def _load_2d_array(path: Path, *, label: str) -> np.ndarray:
    values = squeeze_single_sample(load_array(path))
    if values.ndim != 2:
        raise InputShapeError(
            f"The {label} array must be 2D or a single-sample batch. "
            f"Got shape {values.shape}."
        )
    return values


def _validate_matching_shape(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    if reference.shape != candidate.shape:
        raise InputShapeError(
            "Comparison requires matching shapes. "
            f"Got {reference.shape} and {candidate.shape}."
        )


def _compute_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> FieldComparisonMetrics:
    difference = candidate - reference
    abs_difference = np.abs(difference)
    reference_l2_norm = float(np.linalg.norm(reference))
    difference_l2_norm = float(np.linalg.norm(difference))
    if reference_l2_norm == 0.0:
        relative_l2_error = 0.0 if difference_l2_norm == 0.0 else float("inf")
    else:
        relative_l2_error = difference_l2_norm / reference_l2_norm

    return FieldComparisonMetrics(
        shape=reference.shape,
        reference_dtype=str(reference.dtype),
        candidate_dtype=str(candidate.dtype),
        max_abs_error=float(np.max(abs_difference)),
        mean_abs_error=float(np.mean(abs_difference)),
        rmse=float(np.sqrt(np.mean(abs_difference ** 2))),
        relative_l2_error=relative_l2_error,
        reference_l2_norm=reference_l2_norm,
        passes_tolerance=bool(
            np.allclose(candidate, reference, atol=atol, rtol=rtol)
        ),
        atol=atol,
        rtol=rtol,
    )


__all__ = [
    "ComparisonWorkflowResult",
    "FieldComparisonMetrics",
    "RoundtripValidationResult",
    "compare_field_files",
    "validate_roundtrip",
]
