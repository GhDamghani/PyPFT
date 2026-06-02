"""Focused tests for the legacy-style naive DHT implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from pypft.core.exceptions import GridMismatchError, InputShapeError
from pypft.dht.naive import (
    NaiveDHTImplementation,
    _apply_inverse_dht,
    _as_numpy_with_restorer,
    _build_inverse_kernel,
    _build_weighted_kernel,
)
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid


@pytest.mark.parametrize(
    ("source_radial_size", "target_radial_size", "angular_size", "batch_size"),
    [
        pytest.param(3, 5, 4, 2, id="rectangular-small"),
        pytest.param(7, 9, 8, 2, id="rectangular-larger"),
    ],
)
def test_naive_dht_matches_direct_normalized_midpoint_formula(
    source_radial_size: int,
    target_radial_size: int,
    angular_size: int,
    batch_size: int,
) -> None:
    implementation = NaiveDHTImplementation()
    values = _make_complex_batch(
        batch_size=batch_size,
        radial_size=source_radial_size,
        angular_size=angular_size,
    )
    source_grid = PolarSpatialGrid(
        radial_size=source_radial_size,
        angular_size=angular_size,
    )
    target_grid = PolarFrequencyGrid(
        radial_size=target_radial_size,
        angular_size=angular_size,
    )

    transformed = implementation.apply(
        values,
        source_grid=source_grid,
        target_grid=target_grid,
        direction="forward",
    )

    expected = _normalized_midpoint_direct_dht(
        values,
        target_radial_size=target_grid.radial_size,
    )

    assert transformed.shape == (
        batch_size,
        target_radial_size,
        angular_size,
    )
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


def test_inverse_kernel_warns_when_rcond_truncates_mode() -> None:
    forward_kernel = np.zeros((2, 2, 2), dtype=np.complex128)
    forward_kernel[:, :, 0] = np.array(
        [[1.0, 0.0], [0.0, 1e-14]],
        dtype=np.complex128,
    )
    forward_kernel[:, :, 1] = np.eye(2, dtype=np.complex128)

    with pytest.warns(
        RuntimeWarning,
        match=r"mode 0: retained rank 1/2",
    ):
        inverse_kernel = _build_inverse_kernel(forward_kernel)

    np.testing.assert_allclose(
        inverse_kernel[:, :, 0],
        np.array(
            [[1.0, 0.0], [0.0, 0.0]],
            dtype=np.complex128,
        ),
    )
    np.testing.assert_allclose(
        inverse_kernel[:, :, 1],
        np.eye(2, dtype=np.complex128),
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


def test_apply_inverse_dht_batch_matches_per_image_output(
    sample_batch: np.ndarray,
) -> None:
    source_radial_size = sample_batch.shape[-2]
    angular_size = sample_batch.shape[-1]
    target_radial_size = 5

    transformed = _apply_inverse_dht(
        sample_batch,
        source_radial_size=source_radial_size,
        target_radial_size=target_radial_size,
        angular_size=angular_size,
    )
    expected = np.stack(
        [
            _apply_inverse_dht(
                image,
                source_radial_size=source_radial_size,
                target_radial_size=target_radial_size,
                angular_size=angular_size,
            )
            for image in sample_batch
        ],
        axis=0,
    )

    assert transformed.shape == (sample_batch.shape[0], 5, angular_size)
    np.testing.assert_allclose(transformed, expected)


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
            source_grid=PolarSpatialGrid.infer_from_shape(
                sample_batch.shape[-2:]
            ),
            target_grid=PolarFrequencyGrid(radial_size=5, angular_size=6),
            direction="forward",
        )


def test_as_numpy_with_restorer_detects_cupy_subclasses_by_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCupyArray(np.ndarray):
        pass

    class WrappedCupyArray(FakeCupyArray):
        pass

    class FakeCupyModule:
        ndarray = FakeCupyArray
        complex128 = np.complex128

        @staticmethod
        def asnumpy(values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.complex128)

        @staticmethod
        def asarray(values: np.ndarray, *, dtype: np.dtype) -> WrappedCupyArray:
            return np.asarray(values, dtype=dtype).view(WrappedCupyArray)

    monkeypatch.setattr(
        "pypft.dht.naive._load_optional_cupy",
        lambda: FakeCupyModule,
    )

    values = np.arange(8, dtype=np.float64).reshape(2, 4).view(
        WrappedCupyArray
    )

    assert not type(values).__module__.startswith("cupy")

    values_numpy, restore_output = _as_numpy_with_restorer(values)
    restored = restore_output(values_numpy)

    assert isinstance(restored, FakeCupyArray)
    np.testing.assert_allclose(values_numpy, values)
    np.testing.assert_allclose(restored, values)


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


def _make_complex_batch(
    *,
    batch_size: int,
    radial_size: int,
    angular_size: int,
) -> np.ndarray:
    batch_indices = np.arange(batch_size, dtype=np.float64)[:, None, None]
    radial_indices = np.arange(radial_size, dtype=np.float64)[None, :, None]
    angular_indices = np.arange(angular_size, dtype=np.float64)[
        None,
        None,
        :,
    ]
    real_part = (
        1.0
        + batch_indices
        + 0.25 * radial_indices
        + 0.5 * angular_indices
    )
    imag_part = (
        -0.5 * batch_indices
        + 0.75 * radial_indices
        - 0.25 * angular_indices
    )
    return np.asarray(real_part + 1j * imag_part, dtype=np.complex128)


def _normalized_midpoint_samples(size: int) -> np.ndarray:
    return (0.5 + np.arange(size, dtype=np.float64)) / float(size)
