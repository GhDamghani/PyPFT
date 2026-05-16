"""Tests for the top-level PyPFT package and facade contract.

These checks cover exported package symbols, default configuration, and the
current mock DHT behavior of ``forward`` and ``backward``.
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

    assert pft.config.dht_implementation == "mock-mirror"
    assert pft.config.enable_batching is False
    assert pft.dht_implementation.key == "mock-mirror"


def test_forward_returns_input_for_mock_mirror(
    sample_image: np.ndarray,
) -> None:
    pft = PyPFT()

    transformed = pft.forward(sample_image)

    np.testing.assert_allclose(transformed, sample_image)


def test_backward_returns_input_for_mock_mirror(
    sample_image: np.ndarray,
) -> None:
    pft = PyPFT()

    transformed = pft.backward(sample_image)

    np.testing.assert_allclose(transformed, sample_image)


def test_batched_input_returns_same_values_for_mock_mirror(
    sample_batch: np.ndarray,
) -> None:
    pft = PyPFT(enable_batching=True)

    transformed = pft.forward(sample_batch)

    np.testing.assert_allclose(transformed, sample_batch)
