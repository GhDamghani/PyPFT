"""DHT implementation that builds the Bessel kernel via the order recurrence.

Per the three-term recurrence relation (bessel_properties.md, from Walet's
"Properties of Bessel Functions"):

    J_{k-1}(z) + J_{k+1}(z) = (2k / z) * J_k(z)

i.e. ``J_{k+1}(z) = (2k / z) * J_k(z) - J_{k-1}(z)``, every Bessel function of
integer order can be built up from ``J_0`` and ``J_1`` with cheap elementwise
arithmetic instead of one direct ``scipy.special.jv`` evaluation per order. This
implementation applies that recurrence, still behind the same ``(n, size)``
LRU cache as ``pypft.dht._cached.CachedBesselDHT``.
"""

from functools import lru_cache

import numpy as np
from scipy.special import jn_zeros, jv

from ._cached import CachedBesselDHT


def _bessel_via_recurrence(order: int, z: np.ndarray) -> np.ndarray:
    """Compute ``J_order(z)`` by upward recurrence from ``J_0(z)`` and ``J_1(z)``.

    :param order: The (non-negative integer) Bessel-function order to compute.
    :type order: int
    :param z: The argument(s) to evaluate the Bessel function at.
    :type z: np.ndarray
    :returns: ``J_order(z)``, with the same shape as ``z``.
    :rtype: np.ndarray

    """
    j_prev = jv(0, z)
    if order == 0:
        return j_prev
    j_curr = jv(1, z)
    for k in range(1, order):
        j_prev, j_curr = j_curr, (2 * k / z) * j_curr - j_prev
    return j_curr


def _build_recurrence_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``Y^{nN}`` kernel using the Bessel order recurrence.

    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param size: The length of the vectors the kernel transforms.
    :type size: int
    :returns: The kernel matrix and the ``size + 1`` Bessel zeros it was
        built from.
    :rtype: tuple[np.ndarray, np.ndarray]

    """
    zeros = jn_zeros(n, size + 1)
    j_nN = zeros[-1]
    jn_vals = zeros[:-1]
    outer = np.outer(jn_vals, jn_vals) / j_nN
    j_n_outer = _bessel_via_recurrence(n, outer)
    j_np1_vals = _bessel_via_recurrence(n + 1, jn_vals)
    kernel = (2.0 / j_nN) * j_n_outer / j_np1_vals[np.newaxis, :] ** 2
    return kernel, zeros


@lru_cache(maxsize=None)
def _cached_recurrence_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache ``_build_recurrence_kernel`` by ``(n, size)``.

    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param size: The length of the vectors the kernel transforms.
    :type size: int
    :returns: The kernel matrix and the Bessel zeros it was built from, both
        marked read-only since they are shared across every caller.
    :rtype: tuple[np.ndarray, np.ndarray]

    """
    kernel, zeros = _build_recurrence_kernel(n, size)
    kernel.setflags(write=False)
    zeros.setflags(write=False)
    return kernel, zeros


class RecurrenceBesselDHT(CachedBesselDHT):
    """DHT whose kernel's Bessel values are built via the order recurrence."""

    @staticmethod
    def _bessel_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the LRU-cached, recurrence-built kernel and zeros.

        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param size: The length of the vectors the kernel transforms.
        :type size: int
        :returns: The kernel matrix and the Bessel zeros it was built from.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        return _cached_recurrence_kernel(n, size)
