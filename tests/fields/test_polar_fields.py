from __future__ import annotations

import numpy as np
import pytest

from pypft.core.exceptions import (
    GridMismatchError,
    InputShapeError,
    InvalidFieldOperationError,
)
from pypft.fields.polar import (
    AngularSpectrum,
    FrequencySamples,
    SpatialSamples,
)
from pypft.grids.polar import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarSpatialGrid,
)


def test_spatial_samples_validate_trailing_grid_axes(
    sample_batch: np.ndarray,
) -> None:
    field = SpatialSamples(
        data=sample_batch,
        grid=PolarSpatialGrid.infer_from_shape(sample_batch.shape[-2:]),
    )

    assert field.shape == sample_batch.shape
    assert field.batch_shape == (sample_batch.shape[0],)
    np.testing.assert_allclose(field.asarray(copy=True), sample_batch)


def test_field_rejects_inputs_without_grid_axes() -> None:
    with pytest.raises(InputShapeError):
        SpatialSamples(
            data=np.ones(4, dtype=np.complex128),
            grid=PolarSpatialGrid(radial_size=1, angular_size=4),
        )


def test_field_rejects_mismatched_grid_shape(
    sample_image: np.ndarray,
) -> None:
    with pytest.raises(GridMismatchError):
        SpatialSamples(
            data=sample_image,
            grid=PolarSpatialGrid(radial_size=9, angular_size=9),
        )


def test_same_stage_addition_requires_matching_grid(
    sample_image: np.ndarray,
) -> None:
    grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    lhs = SpatialSamples(data=sample_image, grid=grid)
    rhs = SpatialSamples(data=np.ones_like(sample_image), grid=grid)

    summed = lhs + rhs

    assert isinstance(summed, SpatialSamples)
    np.testing.assert_allclose(summed.asarray(copy=True), sample_image + 1.0)


def test_cross_stage_addition_fails_early(
    sample_image: np.ndarray,
) -> None:
    spatial = SpatialSamples(
        data=sample_image,
        grid=PolarSpatialGrid.infer_from_shape(sample_image.shape),
    )
    frequency = FrequencySamples(
        data=sample_image,
        grid=PolarFrequencyGrid.infer_from_shape(sample_image.shape),
    )

    with pytest.raises(InvalidFieldOperationError):
        _ = spatial + frequency


def test_common_methods_preserve_stage_type(
    sample_image: np.ndarray,
) -> None:
    spectrum = AngularSpectrum(
        data=sample_image,
        grid=PolarAngularModeGrid(
            radial_size=sample_image.shape[0],
            angular_mode_count=sample_image.shape[1],
        ),
    )

    conjugated = spectrum.conj()
    recast = spectrum.astype(np.complex64)

    assert isinstance(conjugated, AngularSpectrum)
    assert isinstance(recast, AngularSpectrum)
    assert recast.asarray().dtype == np.complex64
    assert not spectrum.real.flags.writeable
    assert not spectrum.imag.flags.writeable
    np.testing.assert_allclose(
        conjugated.asarray(copy=True),
        np.conjugate(sample_image),
    )
