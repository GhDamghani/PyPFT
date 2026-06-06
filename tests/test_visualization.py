from __future__ import annotations

from pathlib import Path

import numpy as np

from pypft import PyPFT
from pypft.visualization import (
    FieldRenderSpec,
    normalize_complex_view,
    save_trace_figures,
)


def test_normalize_complex_view_accepts_angular_alias() -> None:
    assert normalize_complex_view("angular") == "phase"


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
