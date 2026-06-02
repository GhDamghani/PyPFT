"""Focused tests for the legacy-style naive DHT implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.dht.naive import NaiveDHTImplementation, _build_weighted_kernel
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


def test_naive_dht_matches_direct_normalized_midpoint_formula(
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

    expected = _normalized_midpoint_direct_dht(
        sample_batch,
        target_radial_size=target_grid.radial_size,
    )

    assert transformed.shape == (2, 5, 4)
    np.testing.assert_allclose(transformed, expected)


def test_weighted_kernel_normalizes_target_radial_samples() -> None:
    kernel = _build_weighted_kernel(
        source_radial_size=2,
        target_radial_size=4,
        angular_size=2,
    )

    rho = _normalized_midpoint_samples(2)
    radii = _normalized_midpoint_samples(4)
    expected = jv(0, np.pi * radii[:, None] * rho[None, :]) * rho[None, :]

    np.testing.assert_allclose(kernel[:, :, 1], expected)


def test_weighted_kernel_requires_even_angular_size() -> None:
    with pytest.raises(
        InputShapeError,
        match="requires an even angular size",
    ):
        _build_weighted_kernel(
            source_radial_size=2,
            target_radial_size=4,
            angular_size=3,
        )


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


def _normalized_midpoint_direct_dht(
    values: np.ndarray,
    *,
    target_radial_size: int,
) -> np.ndarray:
    source_radial_size, angular_size = values.shape[-2:]
    rho = _normalized_midpoint_samples(source_radial_size)
    radii = _normalized_midpoint_samples(target_radial_size)
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


def _normalized_midpoint_samples(size: int) -> np.ndarray:
    return (0.5 + np.arange(size, dtype=np.float64)) / float(size)