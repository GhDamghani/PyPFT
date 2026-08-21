"""Naive, uncached, unoptimized reference DHT implementation."""

import numpy as np
from scipy.special import jn_zeros, jv

from ._base import BaseDHT


class NaiveDHT(BaseDHT):
    """Reference DHT implementation: direct ``scipy.special`` calls, plain matmul.

    Every other implementation is verified against this one and is a strategy
    override of one of its two hooks; this class is never itself overridden.
    """

    @staticmethod
    def _bessel_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Build the ``Y^{nN}`` kernel directly from ``scipy.special.jv`` (Eq. 39).

        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param size: The length of the vectors the kernel transforms.
        :type size: int
        :returns: The kernel matrix and the ``size + 1`` Bessel zeros it was
            built from.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        zeros = jn_zeros(n=n, nt=size + 1)
        j_nN = zeros[-1]
        jn_vals = zeros[:-1]
        j_np1_vals = jv(n + 1, jn_vals)
        outer = np.outer(a=jn_vals, b=jn_vals) / j_nN
        j_n_outer = jv(n, outer)
        kernel = (2.0 / j_nN) * j_n_outer / j_np1_vals[np.newaxis, :] ** 2
        return kernel, zeros

    @staticmethod
    def _apply(kernel: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Apply the kernel via a plain ``numpy`` matrix product.

        :param kernel: The ``(size, size)`` kernel matrix.
        :type kernel: np.ndarray
        :param vector: A length-``size`` signal, or a ``(size, batch)`` stack.
        :type vector: np.ndarray
        :returns: The kernel applied to ``vector``.
        :rtype: np.ndarray

        """
        return kernel @ vector
