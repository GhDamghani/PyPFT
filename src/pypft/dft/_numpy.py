"""Angular DFT implementation using ``numpy.fft``."""

import numpy as np

from ._base import BaseDFT


class NumpyDFT(BaseDFT):
    """Angular DFT computed via ``numpy.fft.fft``/``numpy.fft.ifft``."""

    @staticmethod
    def _forward(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the natural-order forward FFT via ``numpy.fft.fft``.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order FFT of ``values`` along ``axis``.
        :rtype: np.ndarray

        """
        return np.fft.fft(a=values, axis=axis)

    @staticmethod
    def _inverse(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the natural-order inverse FFT via ``numpy.fft.ifft``.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order inverse FFT of ``values`` along ``axis``.
        :rtype: np.ndarray

        """
        return np.fft.ifft(a=values, axis=axis)
