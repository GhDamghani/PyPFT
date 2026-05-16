"""Tests for execution backend selection and optional GPU support.

These checks keep the backend seam honest by verifying CPU behavior, the error
path when the optional GPU dependency is absent, and the current GPU mock-path
parity when CuPy is available.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pypft import PyPFT
from pypft.backends import CPUBackend, create_backend
from pypft.core.exceptions import BackendUnavailableError, UnknownBackendError
from pypft.dft import AngularFFT
from pypft.idft import AngularIFFT


def test_create_backend_returns_cpu_backend() -> None:
    backend = create_backend("cpu")

    assert isinstance(backend, CPUBackend)
    assert backend.key == "cpu"


def test_unknown_backend_key_raises() -> None:
    with pytest.raises(UnknownBackendError):
        create_backend("missing")


def test_gpu_backend_requires_optional_dependency_when_missing() -> None:
    if importlib.util.find_spec("cupy") is not None:
        pytest.skip(
            "CuPy is installed; the missing-dependency path is unavailable."
        )

    with pytest.raises(BackendUnavailableError):
        create_backend("gpu")


@pytest.mark.gpu
def test_gpu_fft_round_trip_matches_identity_if_available(
    sample_image: np.ndarray,
) -> None:
    cupy = pytest.importorskip("cupy")

    backend = create_backend("gpu")
    values = backend.as_complex128(sample_image)
    transformed = AngularFFT(backend=backend).execute(values)
    reconstructed = AngularIFFT(backend=backend).execute(transformed)

    cupy.testing.assert_allclose(reconstructed, values)


@pytest.mark.gpu
def test_gpu_public_api_matches_cpu_for_mock_mirror(
    sample_batch: np.ndarray,
) -> None:
    cupy = pytest.importorskip("cupy")

    cpu_pft = PyPFT(enable_batching=True)
    gpu_pft = PyPFT(enable_batching=True, backend="gpu")

    cpu_output = cpu_pft.forward(sample_batch)
    gpu_output = gpu_pft.forward(sample_batch)

    cupy.testing.assert_allclose(gpu_output, cupy.asarray(cpu_output))
