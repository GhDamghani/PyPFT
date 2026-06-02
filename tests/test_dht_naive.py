"""Focused tests for the legacy-style naive DHT implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.dht.naive import NaiveDHTImplementation
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


def test_naive_dht_matches_direct_legacy_formula(
    sample_batch: np.ndarray,
) -> None:
    implementation = NaiveDHTImplementation()
    source_grid = PolarSpatialGrid.infer_from_shape(sample_batch.shape[-2:])
    target_grid = PolarFrequencyGrid(radial_size=5, angular_size=4)

    transformed = implementation.apply(
        sample_batch,
        source_grid=source_grid,
        target_grid=target_grid,
        direction="forward",
    )

    expected = _legacy_direct_dht(
        sample_batch,
        target_radial_size=target_grid.radial_size,
    )

    assert transformed.shape == (2, 5, 4)
    np.testing.assert_allclose(transformed, expected)


def test_naive_dht_supports_two_dimensional_inputs(
    sample_image: np.ndarray,
) -> None:
    implementation = NaiveDHTImplementation()
    source_grid = PolarFrequencyGrid.infer_from_shape(sample_image.shape)
    target_grid = PolarSpatialGrid(radial_size=4, angular_size=4)

    transformed = implementation.apply(
        sample_image,
        source_grid=source_grid,
        target_grid=target_grid,
        direction="backward",
    )

    assert transformed.shape == (4, 4)


def test_naive_dht_requires_even_angular_size() -> None:
    implementation = NaiveDHTImplementation()
    values = np.ones((1, 3, 3), dtype=np.complex128)

    with pytest.raises(InputShapeError):
        implementation.apply(
            values,
            source_grid=PolarSpatialGrid(radial_size=3, angular_size=3),
            target_grid=PolarFrequencyGrid(radial_size=2, angular_size=3),
            direction="forward",
        )


def test_naive_dht_requires_matching_angular_sizes(
    sample_batch: np.ndarray,
) -> None:
    implementation = NaiveDHTImplementation()

    with pytest.raises(GridMismatchError):
        implementation.apply(
            sample_batch,
            source_grid=PolarSpatialGrid.infer_from_shape(sample_batch.shape[-2:]),
            target_grid=PolarFrequencyGrid(radial_size=5, angular_size=6),
            direction="forward",
        )


def _legacy_direct_dht(
    values: np.ndarray,
    *,
    target_radial_size: int,
) -> np.ndarray:
    source_radial_size, angular_size = values.shape[-2:]
    rho = (0.5 + np.arange(source_radial_size, dtype=np.float64)) / float(
        source_radial_size
    )
    radii = 0.5 * (1.0 + np.arange(target_radial_size, dtype=np.float64))
    orders = np.arange(-(angular_size // 2), angular_size // 2)

    expected = np.zeros(
        (values.shape[0], target_radial_size, angular_size),
        dtype=np.complex128,
    )
    scaled_values = values * rho[None, :, None]

    for mode_index, order in enumerate(orders):
        for radius_index, radius in enumerate(radii):
            weights = jv(order, np.pi * rho * radius)
            expected[:, radius_index, mode_index] = np.sum(
                scaled_values[:, :, mode_index] * weights[None, :],
                axis=1,
            )

    return expected