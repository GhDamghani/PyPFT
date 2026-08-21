"""DHT implementation with an LRU-cached Bessel kernel lookup.

The ``Y^{nN}`` kernel (baddour2019.md, Eq. 39) depends only on the transform's
order ``n`` and size, never on the signal being transformed. Caching it keyed on
``(n, size)`` therefore reuses work across every call at the same order/size,
unlike caching individual ``scipy.special.jv`` evaluations (which take array
arguments and so are unhashable, and would have a low hit rate regardless).
"""

from functools import lru_cache

import numpy as np

from ._naive import NaiveDHT

#: Maximum number of distinct ``(n, size)`` kernels kept alive at once. Each
#: kernel is a ``(size, size)`` ``complex128`` array (~17 MB at ``size=1024``,
#: per the plan's memory estimate), so an unbounded cache risks unbounded
#: growth across a long-running process cycling through many orders/sizes.
KERNEL_CACHE_MAXSIZE = 128


@lru_cache(maxsize=KERNEL_CACHE_MAXSIZE)
def _cached_naive_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache ``NaiveDHT._bessel_kernel`` by ``(n, size)``.

    :param n: The order of the discrete Hankel transform.
    :type n: int
    :param size: The length of the vectors the kernel transforms.
    :type size: int
    :returns: The kernel matrix and the Bessel zeros it was built from, both
        marked read-only since they are shared across every caller.
    :rtype: tuple[np.ndarray, np.ndarray]

    """
    kernel, zeros = NaiveDHT._bessel_kernel(n=n, size=size)
    kernel.setflags(write=False)
    zeros.setflags(write=False)
    return kernel, zeros


class CachedBesselDHT(NaiveDHT):
    """DHT using the naive kernel construction, memoized on ``(n, size)``."""

    @staticmethod
    def _bessel_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the LRU-cached kernel and zeros for ``(n, size)``.

        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param size: The length of the vectors the kernel transforms.
        :type size: int
        :returns: The kernel matrix and the Bessel zeros it was built from.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        return _cached_naive_kernel(n, size)
