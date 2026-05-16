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
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.idft import AngularIFFT
from pypft.operators import BackwardPFTPlan, ForwardPFTPlan


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
        angular_transform=AngularFFT(),
        radial_transform=dht,
        angular_reconstruction=AngularIFFT(),
    )

    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_image.shape)
    frequency_grid = PolarFrequencyGrid.infer_from_shape(sample_image.shape)
    transformed = plan.execute(
        sample_image[None, :, :],
        spatial_grid=spatial_grid,
        frequency_grid=frequency_grid,
    )

    np.testing.assert_allclose(transformed[0], sample_image)
    assert dht.calls == ["forward"]


def test_backward_plan_supports_batches(sample_batch: np.ndarray) -> None:
    dht = IdentityDHT()
    plan = BackwardPFTPlan(
        angular_transform=AngularFFT(),
        radial_transform=dht,
        angular_reconstruction=AngularIFFT(),
    )

    frequency_grid = PolarFrequencyGrid.infer_from_shape(
        sample_batch.shape[-2:]
    )
    spatial_grid = PolarSpatialGrid.infer_from_shape(sample_batch.shape[-2:])
    transformed = plan.execute(
        sample_batch,
        frequency_grid=frequency_grid,
        spatial_grid=spatial_grid,
    )

    np.testing.assert_allclose(transformed, sample_batch)
    assert dht.calls == ["backward"]
