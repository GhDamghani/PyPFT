"""Tests for forward and backward operator-plan orchestration.

This module uses a lightweight identity-style DHT test double so the operator
plans can be validated without a real DHT implementation. The focus here is on
pipeline wiring, direction handling, and batch behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pypft.dft import AngularFFT
from pypft.dht.base import DHTDirection
from pypft.dht.naive import NaiveDHTImplementation
from pypft.fields import AngularSpectrum, FrequencySamples, SpatialSamples
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.idft import AngularIFFT
from pypft.operators import (
    AngularDFT,
    AngularIDFT,
    BackwardPFTPlan,
    ForwardPFTPlan,
    RadialDHT,
)


@dataclass
class IdentityDHT:
    """Test double that preserves values while recording the requested
    direction."""

    key: str = "identity-test"
    description: str = "Identity test double for DHT plan wiring."
    supports_batching: bool = True
    calls: list[DHTDirection] = field(default_factory=list)

    def apply(self, values, *, source_grid, target_grid, direction):
        del source_grid, target_grid
        self.calls.append(direction)
        return values


def test_forward_plan_uses_same_contract(
    sample_image: np.ndarray,
) -> None:
    dht = IdentityDHT()
    plan = ForwardPFTPlan(
        angular_transform=AngularDFT(transform=AngularFFT()),
        radial_transform=RadialDHT(implementation=dht),
        angular_reconstruction=AngularIDFT(transform=AngularIFFT()),
    )

    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    frequency_grid = PolarFrequencyGrid.infer_from_shape(sample_image.shape)
    transformed = plan.execute(
        SpatialSamples(data=sample_image[None, :, :], grid=spatial_grid),
        frequency_grid=frequency_grid,
    )

    assert isinstance(transformed, FrequencySamples)
    np.testing.assert_allclose(transformed.asarray()[0], sample_image)
    assert dht.calls == ["forward"]


def test_backward_plan_supports_batches(sample_batch: np.ndarray) -> None:
    dht = IdentityDHT()
    plan = BackwardPFTPlan(
        angular_transform=AngularDFT(transform=AngularFFT()),
        radial_transform=RadialDHT(implementation=dht),
        angular_reconstruction=AngularIDFT(transform=AngularIFFT()),
    )

    frequency_grid = PolarFrequencyGrid.infer_from_shape(
        sample_batch.shape[-2:]
    )
    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_batch.shape[-2:])
    transformed = plan.execute(
        FrequencySamples(data=sample_batch, grid=frequency_grid),
        spatial_grid=spatial_grid,
    )

    assert isinstance(transformed, SpatialSamples)
    np.testing.assert_allclose(transformed.asarray(), sample_batch)
    assert dht.calls == ["backward"]


def test_pipeline_round_trip_preserves_spatial_samples(
    sample_image: np.ndarray,
) -> None:
    dht = NaiveDHTImplementation()
    forward_plan = ForwardPFTPlan(
        angular_transform=AngularDFT(transform=AngularFFT()),
        radial_transform=RadialDHT(implementation=dht),
        angular_reconstruction=AngularIDFT(transform=AngularIFFT()),
    )
    backward_plan = BackwardPFTPlan(
        angular_transform=AngularDFT(transform=AngularFFT()),
        radial_transform=RadialDHT(implementation=dht),
        angular_reconstruction=AngularIDFT(transform=AngularIFFT()),
    )

    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    frequency_grid = PolarFrequencyGrid.infer_from_shape(sample_image.shape)
    spatial = SpatialSamples(data=sample_image, grid=spatial_grid)

    transformed = forward_plan.execute(
        spatial,
        frequency_grid=frequency_grid,
    )
    reconstructed = backward_plan.execute(
        transformed,
        spatial_grid=spatial_grid,
    )

    assert isinstance(transformed, FrequencySamples)
    assert isinstance(reconstructed, SpatialSamples)
    np.testing.assert_allclose(reconstructed.asarray(), sample_image)


def test_angular_operators_round_trip_stage_wrappers(
    sample_image: np.ndarray,
) -> None:
    spatial = SpatialSamples(
        data=sample_image,
        grid=PolarSpatialGrid.infer_from_shape(sample_image.shape),
    )
    forward = AngularDFT(transform=AngularFFT())
    inverse = AngularIDFT(transform=AngularIFFT())

    angular_modes = forward.apply(spatial)
    reconstructed = inverse.apply(angular_modes)

    assert isinstance(angular_modes, AngularSpectrum)
    assert isinstance(reconstructed, SpatialSamples)
    np.testing.assert_allclose(reconstructed.asarray(), sample_image)
