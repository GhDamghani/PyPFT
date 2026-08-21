"""Axis vocabulary and the centered-angular boundary convention.

``Axis`` names PyPFT's ``(radial, angular[, batch])`` array axes (develop_plan.md
§4). ``_center_angular``/``_uncenter_angular`` implement the one centering
convention every stored array follows on its angular axis (develop_plan.md §3.9):
index ``i`` holds the sample whose angle -- or, once transformed, whose harmonic
-- is ``i - size // 2``, not the "natural" (ascending-from-zero) order that
``numpy.fft`` and ``cv2.warpPolar`` both produce on their own. This module is the
*only* place in ``src/`` allowed to call ``numpy.fft.fftshift``/``ifftshift``;
every other module reorders its angular axis by importing these two functions
instead, so the convention is enforced in one place rather than re-derived at
each boundary.
"""

from enum import IntEnum

import numpy as np


class Axis(IntEnum):
    """Semantic names for PyPFT's array axes.

    ``IntEnum`` because the value *is* the ``numpy`` axis index: passing
    ``Axis.RADIAL`` anywhere a plain ``int`` axis is expected just works.
    ``isinstance(Axis.RADIAL, int)`` is ``True``, so
    ``pypft.utils.validators.IntValidator.type_is_int`` already covers every
    ``axis: Axis | int`` parameter -- such a parameter must not be validated
    with ``EnumValidator.type_is_enum`` first, since that raises on a bare
    ``int``.
    """

    RADIAL = 0
    ANGULAR = 1
    BATCH = 2


DEFAULT_BATCH_AXIS: int = -1
"""The only axis a polar-layer entry point ever defaults.

``-1 == 2 == Axis.BATCH`` for a 3-D ``(radial, angular, batch)`` array, so
"default to the last axis" and "default to the batch axis" coincide exactly
where doing so is unambiguous (develop_plan.md §4). Low-level generic
transforms (e.g. ``pypft.dht``) separately default their own ``axis`` to
``-1`` for a different, purely conventional reason; polar layers never
default a *transform* axis -- ``Axis.RADIAL``/``Axis.ANGULAR`` are always
passed explicitly.
"""


def _center_angular(values: np.ndarray, axis: int) -> np.ndarray:
    """Reorder an angular axis from natural to centered order.

    "Natural" order is what ``cv2.warpPolar`` and an uncentered DFT both
    produce: index ``0`` holds angle/harmonic ``0``, ascending. Centered
    order instead holds angle/harmonic ``i - size // 2`` at index ``i``,
    which is what every array PyPFT stores uses (develop_plan.md §3.9).

    :param values: The array to reorder.
    :type values: np.ndarray
    :param axis: The angular axis of ``values``.
    :type axis: int
    :returns: ``values`` with ``axis`` reordered to centered convention.
    :rtype: np.ndarray

    """
    return np.fft.fftshift(values, axes=axis)


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
    return np.fft.ifftshift(values, axes=axis)
