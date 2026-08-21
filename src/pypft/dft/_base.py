"""Strategy-pattern base class for angular DFT implementations.

Every angular DFT implementation is a subclass of ``BaseDFT`` overriding the
``_forward``/``_inverse`` hooks: the raw, natural-order FFT/IFFT call, with no
opinion on centering. ``forward``/``inverse`` are template methods built on
top of those hooks; they never need to be overridden.

``BaseDFT`` -- not its hooks -- owns the centered-angular convention: every
array PyPFT stores has index ``i`` holding angle/harmonic ``i - size // 2``
on its angular axis, but ``numpy.fft``/``scipy.fft`` both expect and produce
natural order (index ``0`` holds angle/harmonic ``0``, ascending).
``forward``/``inverse`` bridge the two by reordering to natural order
before, and back to centered order after, delegating to the hook, via
``pypft.axes._center_angular``/``_uncenter_angular`` -- the sanctioned way
to reorder an angular axis outside ``pypft.axes`` itself.
"""

import numpy as np

from pypft.axes import _center_angular, _uncenter_angular


class BaseDFT:
    """Base class for pluggable angular DFT implementation strategies.

    All methods are classmethods/staticmethods: a DFT implementation is
    stateless and is never instantiated, matching ``pypft.dht._base.BaseDHT``'s
    convention.
    """

    @staticmethod
    def _forward(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the raw, natural-order forward FFT.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order FFT of ``values`` along ``axis``.
        :rtype: np.ndarray
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @staticmethod
    def _inverse(values: np.ndarray, axis: int) -> np.ndarray:
        """Compute the raw, natural-order inverse FFT.

        :param values: The signal to transform, in natural angular order.
        :type values: np.ndarray
        :param axis: The angular axis of ``values``.
        :type axis: int
        :returns: The natural-order inverse FFT of ``values`` along ``axis``.
        :rtype: np.ndarray
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @classmethod
    def forward(cls, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
        """Compute the centered forward angular DFT.

        :param x: The centered space-domain samples, along ``axis`` of an
            otherwise arbitrary-rank array.
        :type x: np.ndarray
        :param axis: The angular axis of ``x``.
        :type axis: int
        :returns: The centered harmonic-domain coefficients.
        :rtype: np.ndarray

        """
        natural = _uncenter_angular(values=x, axis=axis)
        transformed = cls._forward(values=natural, axis=axis)
        return _center_angular(values=transformed, axis=axis)

    @classmethod
    def inverse(cls, X: np.ndarray, *, axis: int = -1) -> np.ndarray:
        """Compute the centered inverse angular DFT.

        :param X: The centered harmonic-domain coefficients, along ``axis``
            of an otherwise arbitrary-rank array.
        :type X: np.ndarray
        :param axis: The angular axis of ``X``.
        :type axis: int
        :returns: The centered space-domain samples.
        :rtype: np.ndarray

        """
        natural = _uncenter_angular(values=X, axis=axis)
        transformed = cls._inverse(values=natural, axis=axis)
        return _center_angular(values=transformed, axis=axis)
