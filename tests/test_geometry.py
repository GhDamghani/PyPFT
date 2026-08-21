"""Tests for the Cartesian <-> polar bridge (``pypft.geometry``).

Two properties are checked: that the round trip through a uniform polar
resampling approximately recovers a smooth image, and the angular-origin
test that pins down *which* physical angle lands at which centered angular
index -- the piece most likely to silently invert, since it crosses the
centering convention with OpenCV's own angle sign.

OpenCV's polar angle is measured directly on image coordinates, i.e.
``atan2(row - center_y, col - center_x)`` with no ``y``-flip -- so rotating
from the positive-``x`` axis towards the positive-``y`` axis (downward on
screen) is the *positive* direction. That is counter-clockwise in image
coordinates but clockwise as drawn, which is why ``+pi/2`` and ``-pi/2`` are
tested as separate cases: a test set symmetric about 0 cannot see this sign
convention if it happens to be flipped.
"""

import numpy as np
import pytest

from pypft.geometry import cartesian_to_polar, polar_to_cartesian

_IMAGE_SIZE = 256
_N_RADIAL = 128
_N_ANGULAR = 96


def _pixel_grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build centered pixel-offset and radius grids for a square image.

    :param size: The side length of the (square) image, in pixels.
    :type size: int
    :returns: ``dx`` (column offset from center), ``dy`` (row offset from
        center), and the radius ``hypot(dx, dy)``, each ``(size, size)``.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]

    """
    center = size // 2
    rows, cols = np.mgrid[0:size, 0:size]
    dx = (cols - center).astype(np.float64)
    dy = (rows - center).astype(np.float64)
    return dx, dy, np.hypot(dx, dy)


def test_polar_to_cartesian_approximately_inverts_cartesian_to_polar():
    """A smooth Gaussian bump survives the polar round trip, inside the disk.

    ``cv2.warpPolar`` only samples inside the largest inscribed circle, so
    the comparison is restricted to that disk (shrunk slightly to avoid its
    boundary's own interpolation error).

    """
    _, _, r = _pixel_grid(_IMAGE_SIZE)
    sigma = _IMAGE_SIZE / 6.4
    image = np.exp(-(r**2) / (2 * sigma**2))

    polar = cartesian_to_polar(image, _N_RADIAL, _N_ANGULAR)
    reconstructed = polar_to_cartesian(polar, _IMAGE_SIZE, _IMAGE_SIZE)

    max_radius = _IMAGE_SIZE / 2.0
    disk = r < 0.9 * max_radius
    error = np.abs(image[disk] - reconstructed[disk])
    assert error.mean() < 0.02
    assert error.max() < 0.05


#: ``(label, phi0)`` cases in OpenCV's own angle convention (this module's
#: docstring). ``+pi/2``/``-pi/2`` are kept separate -- see the docstring.
_WEDGE_CASES = [
    ("zero", 0.0),
    ("+pi/2", np.pi / 2),
    ("-pi/2", -np.pi / 2),
    ("pi", np.pi),
]


@pytest.mark.parametrize("label,phi0", _WEDGE_CASES, ids=[c[0] for c in _WEDGE_CASES])
def test_angular_origin_wedge_lands_at_the_predicted_centered_index(label, phi0):
    """A Cartesian wedge at OpenCV-angle ``phi0`` lands at its predicted index."""
    dx, dy, r = _pixel_grid(_IMAGE_SIZE)
    cv_angle = np.arctan2(dy, dx)
    angular_distance = (cv_angle - phi0 + np.pi) % (2 * np.pi) - np.pi
    max_radius = _IMAGE_SIZE / 2.0
    wedge = (
        (np.abs(angular_distance) < np.deg2rad(5)) & (r > 5) & (r < 0.9 * max_radius)
    )
    image = wedge.astype(np.float64)

    polar = cartesian_to_polar(image, _N_RADIAL, _N_ANGULAR)
    angular_energy = polar.sum(axis=0)
    measured_index = int(np.argmax(angular_energy))

    predicted_index = (
        _N_ANGULAR // 2 + round(phi0 * _N_ANGULAR / (2 * np.pi))
    ) % _N_ANGULAR
    assert measured_index == predicted_index
