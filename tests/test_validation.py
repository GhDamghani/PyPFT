"""Tests for API-boundary input normalization and grid validation.

The checks in this module focus on shape handling, batch enablement, type
coercion, and explicit-grid compatibility before the transform pipeline reaches
the FFT or DHT stages.
"""

from __future__ import annotations

import numpy as np
import pytest

from pypft import PyPFT
from pypft.backends import CPUBackend
from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.core.validation import normalize_transform_input
from pypft.grids import PolarSpatialGrid


def test_single_image_is_promoted_to_batch(sample_image: np.ndarray) -> None:
    normalized = normalize_transform_input(
        sample_image,
        enable_batching=False,
        backend=CPUBackend(),
    )

    assert normalized.values.shape == (1, 3, 4)
    assert normalized.values.dtype == np.complex128
    assert normalized.had_batch_axis is False


def test_batch_requires_explicit_enablement(sample_batch: np.ndarray) -> None:
    with pytest.raises(InputShapeError):
        normalize_transform_input(
            sample_batch,
            enable_batching=False,
            backend=CPUBackend(),
        )


def test_non_2d_or_3d_input_is_rejected() -> None:
    with pytest.raises(InputShapeError):
        normalize_transform_input(
            np.ones((2, 3, 4, 5)),
            enable_batching=True,
            backend=CPUBackend(),
        )


def test_explicit_spatial_grid_matches_input_shape(
    sample_image: np.ndarray,
) -> None:
    pft = PyPFT(spatial_grid=PolarSpatialGrid(radial_size=9, angular_size=9))

    with pytest.raises(GridMismatchError):
        pft.forward(sample_image)
