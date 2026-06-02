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
from pypft.fields import (
    AngularSpectrum,
    FrequencySamples,
    RadialSpectrum,
    SpatialSamples,
)
from pypft.grids import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarRadialModeGrid,
    PolarSpatialGrid,
)
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


@dataclass
class RecordingDHT:
    key: str = "recording-test"
    description: str = "Recording test double for DHT stage provenance."
    supports_batching: bool = True
    calls: list[DHTDirection] = field(default_factory=list)
    source_grids: list[object] = field(default_factory=list)
    target_grids: list[object] = field(default_factory=list)

    def apply(self, values, *, source_grid, target_grid, direction):
        self.calls.append(direction)
        self.source_grids.append(source_grid)
        self.target_grids.append(target_grid)
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


def test_forward_plan_matches_analytic_gaussian_for_radial_input() -> None:
    radial_size = 32
    angular_size = 8
    gaussian_decay = 12.0
    dht = NaiveDHTImplementation()
    forward_plan = ForwardPFTPlan(
        angular_transform=AngularDFT(transform=AngularFFT()),
        radial_transform=RadialDHT(implementation=dht),
        angular_reconstruction=AngularIDFT(transform=AngularIFFT()),
    )

    radii = _normalized_midpoint_samples(radial_size)
    radial_profile = np.exp(-(gaussian_decay * radii**2))
    spatial = SpatialSamples(
        data=np.tile(radial_profile[:, None], (1, angular_size)),
        grid=PolarSpatialGrid(
            radial_size=radial_size,
            angular_size=angular_size,
        ),
    )

    transformed = forward_plan.execute(
        spatial,
        frequency_grid=PolarFrequencyGrid(
            radial_size=radial_size,
            angular_size=angular_size,
        ),
    )

    expected_radial_profile = (1.0 / (2.0 * gaussian_decay)) * np.exp(
        -((np.pi * radii) ** 2) / (4.0 * gaussian_decay)
    )
    expected = np.tile(expected_radial_profile[:, None], (1, angular_size))

    assert isinstance(transformed, FrequencySamples)
    np.testing.assert_allclose(
        transformed.asarray(),
        expected,
        atol=6e-5,
        rtol=0.0,
    )


def test_angular_operators_round_trip_stage_wrappers(
    sample_image: np.ndarray,
) -> None:
    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    spatial = SpatialSamples(
        data=sample_image,
        grid=spatial_grid,
    )
    forward = AngularDFT(transform=AngularFFT())
    inverse = AngularIDFT(transform=AngularIFFT())

    angular_modes = forward.apply(spatial)
    reconstructed = inverse.apply(angular_modes)

    assert isinstance(angular_modes, AngularSpectrum)
    assert angular_modes.endpoint_grid is spatial_grid
    assert isinstance(reconstructed, SpatialSamples)
    assert reconstructed.grid is spatial_grid
    np.testing.assert_allclose(reconstructed.asarray(), sample_image)


def test_radial_dht_preserves_endpoint_grid_identity(
    sample_image: np.ndarray,
) -> None:
    implementation = RecordingDHT()
    stage = RadialDHT(implementation=implementation)
    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    frequency_grid = PolarFrequencyGrid.infer_from_shape(sample_image.shape)

    angular_modes = AngularSpectrum(
        data=sample_image,
        grid=PolarAngularModeGrid(
            radial_size=sample_image.shape[0],
            angular_mode_count=sample_image.shape[1],
        ),
        endpoint_grid=spatial_grid,
    )
    radial_modes = stage.apply(angular_modes, target_grid=frequency_grid)

    assert isinstance(radial_modes, RadialSpectrum)
    assert implementation.calls == ["forward"]
    assert implementation.source_grids[0] is spatial_grid
    assert implementation.target_grids[0] is frequency_grid
    assert radial_modes.endpoint_grid is frequency_grid

    reconstructed = stage.apply(
        RadialSpectrum(
            data=sample_image,
            grid=PolarRadialModeGrid(
                radial_frequency_size=sample_image.shape[0],
                angular_mode_count=sample_image.shape[1],
            ),
            endpoint_grid=frequency_grid,
        ),
        target_grid=spatial_grid,
    )

    assert isinstance(reconstructed, AngularSpectrum)
    assert implementation.calls == ["forward", "backward"]
    assert implementation.source_grids[1] is frequency_grid
    assert implementation.target_grids[1] is spatial_grid
    assert reconstructed.endpoint_grid is spatial_grid


def _normalized_midpoint_samples(size: int) -> np.ndarray:
    return (0.5 + np.arange(size, dtype=np.float64)) / float(size)
