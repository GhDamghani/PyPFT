"""Angular DFT implementation using ``scipy.fft``."""

from typing import cast

import numpy as np
import scipy.fft

from ._base import BaseDFT


class ScipyDFT(BaseDFT):
    """Angular DFT computed via ``scipy.fft.fft``/``scipy.fft.ifft``."""

    @staticmethod
    def _forward(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the natural-order forward FFT via ``scipy.fft.fft``.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order FFT of ``values`` along ``axis``.
        :rtype: np.ndarray

        """
        # scipy.fft's backend-dispatch decorator confuses static return-type
        # inference (it reports a dispatch-machinery type instead of the
        # array the wrapped function actually returns at runtime).
        return cast(np.ndarray, scipy.fft.fft(values, axis=axis))

    @staticmethod
    def _inverse(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the natural-order inverse FFT via ``scipy.fft.ifft``.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order inverse FFT of ``values`` along ``axis``.
        :rtype: np.ndarray

        """
        return cast(np.ndarray, scipy.fft.ifft(values, axis=axis))
