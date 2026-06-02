from __future__ import annotations

from dataclasses import replace
from typing import Any

from pypft.backends import ExecutionBackend, create_backend
from pypft.config import PyPFTConfig
from pypft.core.validation import (
    normalize_transform_input,
    restore_output_shape,
)
from pypft.dft import AngularFFT
from pypft.dht import DHTImplementation, create_dht_implementation
from pypft.grids import PolarFrequencyGrid, PolarSpatialGrid
from pypft.idft import AngularIFFT
from pypft.operators import BackwardPFTPlan, ForwardPFTPlan


class PyPFT:
    """Public facade for forward and backward polar Fourier transforms."""

    def __init__(
        self,
        *,
        dht_implementation: str = "naive",
        enable_batching: bool = False,
        backend: str = "cpu",
        spatial_grid: PolarSpatialGrid | None = None,
        frequency_grid: PolarFrequencyGrid | None = None,
        config: PyPFTConfig | None = None,
    ) -> None:
        if config is None:
            config = PyPFTConfig(
                dht_implementation=dht_implementation,
                enable_batching=enable_batching,
                backend=backend,
                spatial_grid=spatial_grid,
                frequency_grid=frequency_grid,
            )
        self._config = config
        self._backend: ExecutionBackend = create_backend(config.backend)
        self._dht: DHTImplementation = create_dht_implementation(
            config.dht_implementation
        )
        self._forward_plan = ForwardPFTPlan(
            angular_transform=AngularFFT(backend=self._backend),
            radial_transform=self._dht,
            angular_reconstruction=AngularIFFT(backend=self._backend),
        )
        self._backward_plan = BackwardPFTPlan(
            angular_transform=AngularFFT(backend=self._backend),
            radial_transform=self._dht,
            angular_reconstruction=AngularIFFT(backend=self._backend),
        )

    @property
    def config(self) -> PyPFTConfig:
        return self._config

    @property
    def backend(self) -> ExecutionBackend:
        return self._backend

    @property
    def dht_implementation(self) -> DHTImplementation:
        return self._dht

    def forward(self, image: Any):
        normalized = normalize_transform_input(
            image,
            enable_batching=self._config.enable_batching,
            backend=self._backend,
        )
        spatial_grid = self._resolve_spatial_grid(normalized.values.shape[-2:])
        frequency_grid = self._resolve_frequency_grid(
            normalized.values.shape[-2:]
        )
        transformed = self._forward_plan.execute(
            normalized.values,
            spatial_grid=spatial_grid,
            frequency_grid=frequency_grid,
        )
        return restore_output_shape(
            transformed,
            had_batch_axis=normalized.had_batch_axis,
        )

    def backward(self, image: Any):
        normalized = normalize_transform_input(
            image,
            enable_batching=self._config.enable_batching,
            backend=self._backend,
        )
        frequency_grid = self._resolve_frequency_grid(
            normalized.values.shape[-2:]
        )
        spatial_grid = self._resolve_spatial_grid(normalized.values.shape[-2:])
        transformed = self._backward_plan.execute(
            normalized.values,
            frequency_grid=frequency_grid,
            spatial_grid=spatial_grid,
        )
        return restore_output_shape(
            transformed,
            had_batch_axis=normalized.had_batch_axis,
        )

    def with_dht_implementation(self, key: str) -> "PyPFT":
        return PyPFT(config=replace(self._config, dht_implementation=key))

    def _resolve_spatial_grid(
        self,
        shape: tuple[int, int],
    ) -> PolarSpatialGrid:
        spatial_grid = self._config.spatial_grid
        if spatial_grid is None:
            return PolarSpatialGrid.infer_from_shape(shape)
        spatial_grid.validate_shape(shape)
        return spatial_grid

    def _resolve_frequency_grid(
        self,
        shape: tuple[int, int],
    ) -> PolarFrequencyGrid:
        frequency_grid = self._config.frequency_grid
        if frequency_grid is None:
            return PolarFrequencyGrid.infer_from_shape(shape)
        frequency_grid.validate_shape(shape)
        return frequency_grid


__all__ = ["PyPFT"]
