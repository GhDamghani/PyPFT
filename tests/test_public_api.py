"""Tests for the top-level PyPFT package and facade contract.

These checks cover exported package symbols, default configuration, and the
default naive DHT behavior exposed through ``forward`` and ``backward``.
"""

from __future__ import annotations

import numpy as np

import pypft
from pypft import PyPFT


def test_package_exports() -> None:
    assert "PyPFT" in pypft.__all__
    assert isinstance(pypft.__version__, str)


def test_default_pypft_configuration() -> None:
    pft = PyPFT()

    assert pft.config.dht_implementation == "naive"
    assert pft.config.enable_batching is False
    assert pft.dht_implementation.key == "naive"


def test_forward_preserves_runtime_shape_and_returns_complex128(
    sample_image: np.ndarray,
) -> None:
    pft = PyPFT()

    transformed = pft.forward(sample_image)

    assert transformed.shape == sample_image.shape
    assert transformed.dtype == np.complex128
    assert not np.allclose(transformed, sample_image)


def test_backward_inverts_forward_for_default_dht(
    sample_image: np.ndarray,
) -> None:
    pft = PyPFT()

    forward_transformed = pft.forward(sample_image)
    backward_transformed = pft.backward(forward_transformed)

    np.testing.assert_allclose(backward_transformed, sample_image)


def test_batched_input_matches_per_image_forward_calls(
    sample_batch: np.ndarray,
) -> None:
    pft = PyPFT(enable_batching=True)

    transformed = pft.forward(sample_batch)
    expected = np.stack(
        [PyPFT().forward(image) for image in sample_batch],
        axis=0,
    )

    np.testing.assert_allclose(transformed, expected)
