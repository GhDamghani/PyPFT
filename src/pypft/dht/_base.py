"""Strategy-pattern base class for discrete Hankel transform (DHT) implementations.

Every DHT implementation is a subclass of ``BaseDHT`` that overrides one or both
of two independent hooks, matching the two-axis optimization split of the DHT
computation:

    - ``_bessel_kernel(n, size)``: computes the ``(size, size)`` transform kernel and
        the Bessel zeros it was built from. This is the "computing Bessel values" axis.
    - ``_apply(kernel, vector)``: applies the kernel to a signal (or a batch of signals,
        stacked column-wise). This is the "computing the transformation" axis.

``forward``/``inverse`` are template methods built on top of those two hooks; they
never need to be overridden.

The kernel used here is Baddour's ``Y^{nN}`` formulation (baddour2019.md, Eq. 39): it
is self-inverse (``Y^{nN} Y^{nN} = I``, Eq. 41) even though it is not symmetric, so
forward and inverse share the exact same matrix application and differ only by a
scalar prefactor (Eqs. 42-43), which halves the implementation surface. (The
alternative ``T^{nN}`` formulation of Eq. 44 is also self-inverse and additionally
symmetric, but it acts on the "scaled" vectors of Sec. 7 rather than on ``f``/``F``
directly, so ``Y^{nN}`` is used here to keep the public API in terms of raw signal
values.)
"""

import numpy as np


class BaseDHT:
    """Base class for pluggable DHT implementation strategies.

    All methods are classmethods/staticmethods: a DHT implementation is stateless
    and is never instantiated, matching the rest of the codebase's validator
    convention of stateless methods.
    """

    @staticmethod
    def _bessel_kernel(n: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute the ``(size, size)`` transform kernel and the Bessel zeros.

        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param size: The length of the vectors the kernel transforms.
        :type size: int
        :returns: The ``Y^{nN}`` kernel matrix (Eq. 39) and the ``size + 1`` Bessel
            zeros ``j_n1, ..., j_nN`` it was built from, with ``j_nN = zeros[-1]``.
        :rtype: tuple[np.ndarray, np.ndarray]
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @staticmethod
    def _apply(kernel: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Apply a transform kernel to a signal (or batch of signals).

        :param kernel: The ``(size, size)`` kernel matrix from ``_bessel_kernel``.
        :type kernel: np.ndarray
        :param vector: A length-``size`` signal, or a ``(size, batch)`` stack of
            signals to transform together.
        :type vector: np.ndarray
        :returns: The kernel applied to ``vector``.
        :rtype: np.ndarray
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @classmethod
    def sample_points(
        cls, n: int, size: int, R: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the discretization's space- and frequency-domain sample points.

        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param size: The number of samples (``N - 1`` in baddour2019.md's notation).
        :type size: int
        :param R: The space-domain limit the signal is assumed confined to.
        :type R: float
        :returns: The sample points ``r_nk`` and ``rho_nk`` (Eq. 31), each of
            length ``size``.
        :rtype: tuple[np.ndarray, np.ndarray]

        """
        _, zeros = cls._bessel_kernel(n, size)
        j_nN = zeros[-1]
        jn_vals = zeros[:-1]
        r = jn_vals * R / j_nN
        rho = jn_vals / R
        return r, rho

    @classmethod
    def forward(cls, f: np.ndarray, n: int, R: float) -> np.ndarray:
        """Compute the physical, ``R``-scaled forward discrete Hankel transform.

        :param f: The space-domain samples ``f(r_nk)``, sampled at the points
            returned by ``sample_points``.
        :type f: np.ndarray
        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param R: The space-domain limit ``f`` is assumed confined to.
        :type R: float
        :returns: The frequency-domain samples ``F(rho_nk)`` (Eq. 42).
        :rtype: np.ndarray

        """
        kernel, zeros = cls._bessel_kernel(n, f.shape[0])
        j_nN = zeros[-1]
        return (R**2 / j_nN) * cls._apply(kernel, f)

    @classmethod
    def inverse(cls, F: np.ndarray, n: int, R: float) -> np.ndarray:
        """Compute the physical, ``R``-scaled inverse discrete Hankel transform.

        :param F: The frequency-domain samples ``F(rho_nk)``.
        :type F: np.ndarray
        :param n: The order of the discrete Hankel transform.
        :type n: int
        :param R: The space-domain limit the reconstructed signal is confined to.
        :type R: float
        :returns: The space-domain samples ``f(r_nk)`` (Eq. 43).
        :rtype: np.ndarray

        """
        kernel, zeros = cls._bessel_kernel(n, F.shape[0])
        j_nN = zeros[-1]
        return (j_nN / R**2) * cls._apply(kernel, F)
