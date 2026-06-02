from __future__ import annotations

import pytest

from pypft.core.exceptions import GridMismatchError
from pypft.grids import (
    PolarAngularModeGrid,
    PolarFrequencyGrid,
    PolarRadialModeGrid,
    PolarSpatialGrid,
    PolarTransformGrids,
)


def test_transform_grids_infer_mode_grids_from_endpoints() -> None:
    spatial = PolarSpatialGrid(radial_size=6, angular_size=8)
    frequency = PolarFrequencyGrid(radial_size=10, angular_size=8)

    grids = PolarTransformGrids.from_endpoint_grids(spatial, frequency)

    assert grids.spatial is spatial
    assert grids.frequency is frequency
    assert grids.angular_modes == PolarAngularModeGrid(
        radial_size=6,
        angular_mode_count=8,
    )
    assert grids.radial_modes == PolarRadialModeGrid(
        radial_frequency_size=10,
        angular_mode_count=8,
    )
    assert grids.angular_size == 8
    assert grids.radial_size == 6
    assert grids.radial_frequency_size == 10


def test_transform_grids_require_matching_endpoint_angular_sizes() -> None:
    spatial = PolarSpatialGrid(radial_size=6, angular_size=8)
    frequency = PolarFrequencyGrid(radial_size=10, angular_size=6)

    with pytest.raises(GridMismatchError):
        PolarTransformGrids.from_endpoint_grids(spatial, frequency)