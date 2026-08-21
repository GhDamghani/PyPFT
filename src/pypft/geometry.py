"""The Cartesian <-> polar bridge for ordinary (uniformly-sampled) images.

``cartesian_to_polar``/``polar_to_cartesian`` wrap ``cv2.warpPolar`` (forward and
inverse) so a plain image can be viewed on a polar grid and back, at whatever
resolution the caller asks for. **This is not the transform's own sampling
grid**: ``cv2.warpPolar`` produces a *uniformly* spaced radial axis, whereas the
discrete Hankel transform's grid is order-dependent and non-uniform (Baddour's
``r_nk``, develop_plan.md §3.5); ``pypft.grid.sample_cartesian`` will be the
production sampler once it exists. These two functions exist because
``warpPolar`` is the natural first illustration of what "polar" means for an
image, not because it feeds the transform.

Two more things happen at this boundary, once per direction:

- **Layout.** ``cv2.warpPolar`` lays its polar array out ``(angular, radial[,
  channel])`` -- the reference implementation's convention (develop_plan.md
  §3.10). PyPFT's own convention is the opposite, ``(radial, angular[,
  channel])`` (``pypft.axes.Axis``), so every crossing of this boundary
  transposes deliberately.
- **Angular convention.** ``warpPolar``'s angular axis is in natural
  (ascending-from-zero) order; PyPFT's own convention is centered
  (``pypft.axes._center_angular``, develop_plan.md §3.9). ``cartesian_to_polar``
  centers on the way in, ``polar_to_cartesian`` un-centers on the way out, so
  nothing downstream of this module has to think about it again.
"""

import cv2
import numpy as np

from pypft.axes import Axis, _center_angular, _uncenter_angular
from pypft.utils.validators import IntValidator, NumpyValidator

#: ``cv2.warpPolar``'s interpolation mode: a straight (non-logarithmic) radial
#: axis, matching the DHT's own linearly-spaced sample points.
_WARP_POLAR_FLAGS = cv2.WARP_POLAR_LINEAR


def _center_and_max_radius(
    height: int, width: int
) -> tuple[tuple[float, float], float]:
    """Compute the image center and the largest fully-inscribed radius.

    :param height: The image height, in pixels.
    :type height: int
    :param width: The image width, in pixels.
    :type width: int
    :returns: The ``(x, y)`` center and the radius of the largest circle that
        fits inside the image.
    :rtype: tuple[tuple[float, float], float]

    """
    center = (width / 2.0, height / 2.0)
    max_radius = min(height, width) / 2.0
    return center, max_radius


def cartesian_to_polar(image: np.ndarray, n_radial: int, n_angular: int) -> np.ndarray:
    """Resample a Cartesian image onto a uniform polar grid.

    :param image: A ``(height, width[, channel])`` Cartesian image.
    :type image: np.ndarray
    :param n_radial: The number of (uniformly spaced) radial samples.
    :type n_radial: int
    :param n_angular: The number of angular samples.
    :type n_angular: int
    :returns: A ``(n_radial, n_angular[, channel])`` polar-resampled image, on
        PyPFT's ``(radial, angular)`` layout with a centered angular axis.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(image)
    NumpyValidator.value_is_at_least_1d(image)
    IntValidator.type_is_int(n_radial)
    IntValidator.value_is_positive(n_radial)
    IntValidator.type_is_int(n_angular)
    IntValidator.value_is_positive(n_angular)

    height, width = image.shape[:2]
    center, max_radius = _center_and_max_radius(height, width)

    # warpPolar's dsize is (width, height) and it lays its own output out
    # (angular, radial[, channel]) -- moveaxis undoes that to PyPFT's layout.
    warped = cv2.warpPolar(
        image, (n_radial, n_angular), center, max_radius, _WARP_POLAR_FLAGS
    )
    polar = np.moveaxis(warped, 0, 1)
    return _center_angular(polar, axis=Axis.ANGULAR)


def polar_to_cartesian(polar: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample a uniform polar image back onto a Cartesian grid.

    The exact inverse of ``cartesian_to_polar`` when given the source image's
    own ``height``/``width`` -- both the layout transpose and the angular
    centering are undone before delegating to ``cv2.warpPolar``'s inverse map.

    :param polar: A ``(n_radial, n_angular[, channel])`` polar image, on
        PyPFT's ``(radial, angular)`` layout with a centered angular axis
        (i.e. as produced by ``cartesian_to_polar``).
    :type polar: np.ndarray
    :param height: The desired output image height, in pixels.
    :type height: int
    :param width: The desired output image width, in pixels.
    :type width: int
    :returns: A ``(height, width[, channel])`` Cartesian image.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(polar)
    NumpyValidator.value_is_at_least_1d(polar)
    IntValidator.type_is_int(height)
    IntValidator.value_is_positive(height)
    IntValidator.type_is_int(width)
    IntValidator.value_is_positive(width)

    center, max_radius = _center_and_max_radius(height, width)

    uncentered = _uncenter_angular(polar, axis=Axis.ANGULAR)
    warped = np.moveaxis(uncentered, 0, 1)  # back to warpPolar's own layout
    return cv2.warpPolar(
        warped,
        (width, height),
        center,
        max_radius,
        _WARP_POLAR_FLAGS | cv2.WARP_INVERSE_MAP,
    )
