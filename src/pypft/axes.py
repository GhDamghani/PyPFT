"""Axis vocabulary and the centered-angular boundary convention.

``PolarAxis`` names the two sample axes of a polar array, ``(radial,
angular)``; a batch of samples adds one more axis after them, ``(radial,
angular, batch)``. The batch axis is not a coordinate of the sample, so it has no
member of its own: it is always the axis after the sample axes, named only by
``DEFAULT_BATCH_AXIS``. ``POLAR_SAMPLE_NDIM`` is the rank of a single polar
sample, which is what tells a sample from a batch; ``_value_is_polar_sample_or_batch``
is the one rank check every polar-layer entry point shares.

``_center_angular``/``_uncenter_angular`` implement the one centering
convention every stored array follows on its angular axis: index ``i`` holds
the sample whose angle -- or, once transformed, whose harmonic -- is
``i - size // 2``, not the "natural" (ascending-from-zero) order that
``numpy.fft`` and ``cv2.warpPolar`` both produce on their own. This module is the
*only* place in ``src/`` allowed to call ``numpy.fft.fftshift``/``ifftshift``;
every other module reorders its angular axis by importing these two functions
instead, so the convention is enforced in one place rather than re-derived at
each boundary.
"""

from enum import IntEnum

import numpy as np

from pypft.utils.validators import NumpyValidator


class PolarAxis(IntEnum):
    """Semantic names for the sample axes of a polar array.

    ``IntEnum`` because the value *is* the ``numpy`` axis index: passing
    ``PolarAxis.RADIAL`` anywhere a plain ``int`` axis is expected just works.
    ``isinstance(PolarAxis.RADIAL, int)`` is ``True``, so
    ``pypft.utils.validators.IntValidator.type_is_int`` already covers every
    ``axis: PolarAxis | int`` parameter -- such a parameter must not be validated
    with ``EnumValidator.type_is_enum`` first, since that raises on a bare
    ``int``.

    There is deliberately no batch member: the batch axis is the axis after the
    sample axes, ``DEFAULT_BATCH_AXIS``.
    """

    RADIAL = 0
    ANGULAR = 1


POLAR_SAMPLE_NDIM: int = len(PolarAxis)
"""The rank of a single polar sample, ``(radial, angular)``.

A batch of samples has rank ``POLAR_SAMPLE_NDIM + 1``, ``(radial, angular,
batch)``, and its batch axis sits at index ``POLAR_SAMPLE_NDIM``.
"""

DEFAULT_BATCH_AXIS: int = -1
"""The only axis a polar-layer entry point ever defaults.

The batch axis is the axis after the sample axes, which on a batch of polar
samples is also the last axis, so "default to the last axis" and "default to the
batch axis" coincide. Low-level generic transforms (e.g. ``pypft.dht``)
separately default their own ``axis`` to ``-1`` for a different, purely
conventional reason; polar layers never default a *transform* axis --
``PolarAxis.RADIAL``/``PolarAxis.ANGULAR`` are always passed explicitly.
"""


def _value_is_polar_sample_or_batch(value: np.ndarray) -> None:
    """Value-validator for a single polar sample or a batch of them.

    :param value: The value to be validated.
    :type value: np.ndarray
    :raises ValueError: If ``value`` is neither a single sample ``(radial,
        angular)`` nor a batch ``(radial, angular, batch)``.

    """
    try:
        NumpyValidator.value_has_ndim_in(
            value=value, ndims=(POLAR_SAMPLE_NDIM, POLAR_SAMPLE_NDIM + 1)
        )
    except ValueError as error:
        raise ValueError(
            "value must be a single sample (radial, angular) or a batch (radial, "
            f"angular, batch), got a {value.ndim}-D array"
        ) from error


def _center_angular(values: np.ndarray, axis: int) -> np.ndarray:
    """Reorder an angular axis from natural to centered order.

    "Natural" order is what ``cv2.warpPolar`` and an uncentered DFT both
    produce: index ``0`` holds angle/harmonic ``0``, ascending. Centered
    order instead holds angle/harmonic ``i - size // 2`` at index ``i``,
    which is what every array PyPFT stores uses.

    :param values: The array to reorder.
    :type values: np.ndarray
    :param axis: The angular axis of ``values``.
    :type axis: int
    :returns: ``values`` with ``axis`` reordered to centered convention.
    :rtype: np.ndarray

    """
    return np.fft.fftshift(x=values, axes=axis)


def _uncenter_angular(values: np.ndarray, axis: int) -> np.ndarray:
    """Reorder an angular axis from centered back to natural order.

    The exact inverse of ``_center_angular`` -- see its docstring for what
    "natural" and "centered" mean here.

    :param values: The array to reorder.
    :type values: np.ndarray
    :param axis: The angular axis of ``values``.
    :type axis: int
    :returns: ``values`` with ``axis`` reordered to natural convention.
    :rtype: np.ndarray

    """
    return np.fft.ifftshift(x=values, axes=axis)
