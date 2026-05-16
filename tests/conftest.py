"""Shared pytest fixtures for reusable sample images and batches.

This file centralizes small complex-valued arrays used across the suite so
test inputs stay consistent while individual test modules focus on one layer
of behavior at a time.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a small 2D complex128 image used throughout the test suite."""

    return np.array(
        [
            [1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j, 4.0 + 0.5j],
            [0.5 - 1.0j, 1.5 + 0.5j, 2.5 + 1.5j, 3.5 - 0.5j],
            [2.0 + 2.0j, 1.0 - 1.0j, 0.5 + 0.25j, 4.5 + 0.0j],
        ],
        dtype=np.complex128,
    )


@pytest.fixture
def sample_batch(sample_image: np.ndarray) -> np.ndarray:
    """Return a small leading-axis batch built from the shared sample image."""

    second_image = sample_image * (1.0 + 0.5j)
    return np.stack([sample_image, second_image], axis=0)
