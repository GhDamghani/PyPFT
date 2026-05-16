"""Tests for the angular FFT and IFFT wrapper stages.

These tests compare the PyPFT stage wrappers directly against ``scipy.fft`` so
the angular transform layer is verified independently of the larger PFT
pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as scipy_fft

from pypft.dft import AngularFFT
from pypft.idft import AngularIFFT


def test_angular_fft_matches_scipy(sample_image: np.ndarray) -> None:
    stage = AngularFFT()

    transformed = stage.execute(sample_image)

    np.testing.assert_allclose(
        transformed,
        scipy_fft.fft(sample_image, axis=-1),
    )


def test_angular_ifft_matches_scipy(sample_image: np.ndarray) -> None:
    stage = AngularIFFT()

    reconstructed = stage.execute(sample_image)

    np.testing.assert_allclose(
        reconstructed,
        scipy_fft.ifft(sample_image, axis=-1),
    )
