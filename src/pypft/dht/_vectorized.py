"""DHT implementation with a parallelized kernel-application step.

Inherits ``pypft.dht._recurrence.RecurrenceBesselDHT``'s optimized Bessel kernel
unchanged and overrides only ``_apply`` (the "computing the transformation with
the Bessel values provided" axis), using ``numba``-parallelized loops instead of
relying on NumPy/BLAS's own internal parallelism.
"""

import numpy as np
from numba import njit, prange

from ._recurrence import RecurrenceBesselDHT


@njit(parallel=True, cache=True)
def _matvec_parallel(kernel: np.ndarray, vector: np.ndarray) -> np.ndarray:
    size = kernel.shape[0]
    result = np.empty(size, dtype=np.complex128)
    for i in prange(size):
        acc = 0.0 + 0.0j
        for j in range(size):
            acc += kernel[i, j] * vector[j]
        result[i] = acc
    return result


@njit(parallel=True, cache=True)
def _matmat_parallel(kernel: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    size, batch = matrix.shape
    result = np.empty((size, batch), dtype=np.complex128)
    for i in prange(size):
        for b in range(batch):
            acc = 0.0 + 0.0j
            for j in range(size):
                acc += kernel[i, j] * matrix[j, b]
            result[i, b] = acc
    return result


class VectorizedDHT(RecurrenceBesselDHT):
    """DHT applying its (recurrence-built, cached) kernel via parallel loops."""

    @staticmethod
    def _apply(kernel: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Apply the kernel via a ``numba``-parallelized loop.

        :param kernel: The ``(size, size)`` kernel matrix.
        :type kernel: np.ndarray
        :param vector: A length-``size`` signal, or a ``(size, batch)`` stack.
        :type vector: np.ndarray
        :returns: The kernel applied to ``vector``, cast back to ``vector``'s
            own dtype (real signals stay real).
        :rtype: np.ndarray

        """
        vector_c = vector.astype(np.complex128, copy=False)
        if vector.ndim == 1:
            result = _matvec_parallel(kernel.astype(np.complex128), vector_c)
        else:
            result = _matmat_parallel(kernel.astype(np.complex128), vector_c)
        if not np.iscomplexobj(vector):
            result = np.real(result)
        return result.astype(vector.dtype, copy=False)
