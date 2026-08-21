"""DHT implementation with a parallelized kernel-application step.

Inherits ``pypft.dht._cached.CachedBesselDHT``'s cached Bessel kernel unchanged
and overrides only ``_apply`` (the "computing the transformation with the
Bessel values provided" axis), using ``numba``-parallelized loops instead of
relying on NumPy/BLAS's own internal parallelism.
"""

import numpy as np
from numba import njit, prange

from ._cached import CachedBesselDHT


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


def _apply_batched(kernel: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply ``_matmat_parallel`` to a ``(size, ...)``-shaped, rank > 2 array.

    ``_matmat_parallel`` only understands a plain ``(size, batch)`` matrix, but
    ``BaseDHT._apply_along_axis`` hands ``_apply`` an array whose target axis
    sits at ``-2`` with an arbitrary number of leading dimensions and exactly
    one trailing one. This flattens every dimension
    except the size axis into a single batch dimension, delegates, then
    restores the original shape -- the numba-loop equivalent of the implicit
    broadcasting ``numpy.matmul`` already gives the other implementations.

    :param kernel: The ``(size, size)`` complex kernel matrix.
    :type kernel: np.ndarray
    :param values: A complex array of shape ``(..., size, trailing)``.
    :type values: np.ndarray
    :returns: The kernel applied along the ``size`` axis, same shape as
        ``values``.
    :rtype: np.ndarray

    """
    size = values.shape[-2]
    moved = np.moveaxis(a=values, source=-2, destination=0)  # (size, *rest)
    rest_shape = moved.shape[1:]
    flat = np.ascontiguousarray(moved.reshape(size, -1))
    result_flat = _matmat_parallel(kernel=kernel, matrix=flat)
    result = result_flat.reshape((size,) + rest_shape)
    return np.moveaxis(a=result, source=0, destination=-2)


class VectorizedDHT(CachedBesselDHT):
    """DHT applying its (naive-built, cached) kernel via parallel loops."""

    @staticmethod
    def _apply(kernel: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Apply the kernel via a ``numba``-parallelized loop.

        :param kernel: The ``(size, size)`` kernel matrix.
        :type kernel: np.ndarray
        :param vector: A length-``size`` signal, or an array with a
            length-``size`` dimension at axis ``-2`` (1-D or higher rank).
        :type vector: np.ndarray
        :returns: The kernel applied to ``vector``, cast back to ``vector``'s
            own dtype (real signals stay real).
        :rtype: np.ndarray

        """
        vector_c = vector.astype(np.complex128, copy=False)
        kernel_c = kernel.astype(np.complex128)
        if vector.ndim == 1:
            result = _matvec_parallel(kernel=kernel_c, vector=vector_c)
        elif vector.ndim == 2:
            result = _matmat_parallel(kernel=kernel_c, matrix=vector_c)
        else:
            result = _apply_batched(kernel=kernel_c, values=vector_c)
        if not np.iscomplexobj(vector):
            result = np.real(result)
        return result.astype(vector.dtype, copy=False)
