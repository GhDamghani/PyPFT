"""Shared pytest fixtures for reusable sample images and batches.

This file centralizes small complex-valued arrays used across the suite so
test inputs stay consistent while individual test modules focus on one layer
of behavior at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a deterministic 64x64 complex128 image used throughout tests."""

    radial = np.linspace(0.0, 1.0, 64, dtype=np.float64)[:, None]
    angular = np.linspace(
        0.0,
        2.0 * np.pi,
        64,
        endpoint=False,
        dtype=np.float64,
    )[None, :]
    real = np.sin(2.5 * angular) + radial**2
    imag = np.cos(1.5 * angular) - 0.5 * radial
    return (real + 1j * imag).astype(np.complex128)


@pytest.fixture
def roundtrip_image() -> np.ndarray:
    """Return a compact Lena-derived reference image for exact roundtrip tests."""

    from PIL import Image

    image_path = Path(__file__).with_name("samples") / "lena.tif"
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)
    row_indices = np.linspace(0, image.shape[0] - 1, 3).round().astype(int)
    column_indices = np.linspace(0, image.shape[1] - 1, 4).round().astype(int)
    return image[np.ix_(row_indices, column_indices)].astype(np.complex128)


@pytest.fixture
def sample_batch(sample_image: np.ndarray) -> np.ndarray:
    """Return a small leading-axis batch built from the shared sample image."""

    second_image = sample_image * (1.0 + 0.5j)
    return np.stack([sample_image, second_image], axis=0)
