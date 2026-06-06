from __future__ import annotations

from pathlib import Path

import numpy as np

from pypft import PyPFT
from pypft.visualization.matplotlib_backend import (
    _axis_labels,
    _render_payload,
)
from pypft.visualization import (
    FieldRenderSpec,
    apply_gamma,
    normalize_complex_view,
    save_trace_figures,
)


def test_normalize_complex_view_accepts_angular_alias() -> None:
    assert normalize_complex_view("angular") == "phase"


def test_apply_gamma_normalizes_before_power_law() -> None:
    values = np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float64)

    normalized = apply_gamma(values, gamma=1.0)
    gamma_corrected = apply_gamma(values, gamma=0.5)

    np.testing.assert_allclose(normalized, values / 8.0)
    np.testing.assert_allclose(gamma_corrected, np.sqrt(values / 8.0))


def test_apply_gamma_is_scale_invariant() -> None:
    values = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float64)

    np.testing.assert_allclose(
        apply_gamma(values, gamma=0.5),
        apply_gamma(values * 100.0, gamma=0.5),
    )


def test_frequency_samples_axis_labels_use_frequency_domain_terms(
    sample_image: np.ndarray,
) -> None:
    frame = PyPFT().forward_with_trace(sample_image).trace.final_frame

    assert _axis_labels(frame) == (
        "angular frequency index",
        "radial frequency index",
    )


def test_render_payload_uses_diverging_cmap_for_signed_real_data() -> None:
    data = np.array([[-2.0, -0.5], [1.0, 3.0]], dtype=np.float64)

    rendered, cmap, label, color_limits = _render_payload(
        data,
        view="real",
        gamma=1.0,
    )

    np.testing.assert_allclose(rendered, data)
    assert cmap == "RdBu_r"
    assert label == "value"
    assert color_limits == (-3.0, 3.0)


def test_save_trace_figures_writes_final_stage_views(
    tmp_path: Path,
    sample_image: np.ndarray,
) -> None:
    result = PyPFT().forward_with_trace(sample_image)

    figure_paths = save_trace_figures(
        result.trace,
        tmp_path,
        render_spec=FieldRenderSpec(complex_view="both", gamma=0.5),
        save_all_views=False,
    )

    assert "frequency_samples" in figure_paths
    assert len(figure_paths["frequency_samples"]) == 2
    for path in figure_paths["frequency_samples"]:
        assert path.exists()
