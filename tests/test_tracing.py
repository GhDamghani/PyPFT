from __future__ import annotations

import numpy as np

from pypft import PyPFT


def test_forward_trace_exposes_canonical_published_stages(
    sample_image: np.ndarray,
) -> None:
    result = PyPFT().forward_with_trace(sample_image)

    assert result.trace.direction == "forward"
    assert result.trace.stage_names == (
        "spatial_samples",
        "angular_dft",
        "radial_dht",
        "angular_idft",
        "frequency_samples",
    )
    normalized_output = result.trace.final_frame.asarray()
    if not result.had_batch_axis:
        normalized_output = normalized_output[0]
    np.testing.assert_allclose(result.output, normalized_output)