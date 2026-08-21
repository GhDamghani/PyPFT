"""Shared fixtures for tests exercising the PFT/IPFT pipeline against oracles.

The Gaussian oracle is the exact, known space<->frequency Hankel-transform
pair this package's DHT was already validated against
(``tests/dht/test_gaussian.py``), reused here as the PFT's own end-to-end
accuracy oracle (PeerJ CS Part II's own worked example). The Shepp-Logan
phantom is a standard, code-generated (no binary asset) piecewise-constant
test image, used only for a qualitative round-trip sanity check since it is
not circularly symmetric like the Gaussian.
"""

import numpy as np


def gaussian_f(r: np.ndarray) -> np.ndarray:
    """Compute the space-domain Gaussian oracle ``f(r) = exp(-r^2)``.

    :param r: The radial sample points.
    :type r: np.ndarray
    :returns: ``f`` evaluated at ``r``.
    :rtype: np.ndarray

    """
    return np.exp(-(r**2))


def gaussian_F(rho: np.ndarray) -> np.ndarray:
    """Compute the frequency-domain Gaussian oracle ``F(rho) = pi*exp(-rho^2/4)``.

    The continuous 2-D Fourier transform of ``gaussian_f``'s radially
    symmetric function, per PeerJ CS Part II's own worked example.

    :param rho: The radial frequency sample points.
    :type rho: np.ndarray
    :returns: ``F`` evaluated at ``rho``.
    :rtype: np.ndarray

    """
    return np.pi * np.exp(-(rho**2) / 4.0)


#: Shepp-Logan phantom ellipse parameters, ``(amplitude, a, b, x0, y0,
#: phi_degrees)`` -- the standard parameters (Shepp & Logan, 1974), each
#: ellipse defined on the unit disc.
_SHEPP_LOGAN_ELLIPSES = (
    (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
    (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0),
    (-0.2, 0.11, 0.31, 0.22, 0.0, -18.0),
    (-0.2, 0.16, 0.41, -0.22, 0.0, 18.0),
    (0.1, 0.21, 0.25, 0.0, 0.35, 0.0),
    (0.1, 0.046, 0.046, 0.0, 0.1, 0.0),
    (0.1, 0.046, 0.046, 0.0, -0.1, 0.0),
    (0.1, 0.046, 0.023, -0.08, -0.605, 0.0),
    (0.1, 0.023, 0.023, 0.0, -0.606, 0.0),
    (0.1, 0.023, 0.046, 0.06, -0.605, 0.0),
)


def shepp_logan_phantom(size: int) -> np.ndarray:
    """Generate the standard Shepp-Logan phantom, code-generated (no binary asset).

    :param size: The phantom's height and width, in pixels.
    :type size: int
    :returns: A ``(size, size)`` ``float64`` grayscale phantom on the unit disc.
    :rtype: np.ndarray

    """
    coords = np.linspace(start=-1.0, stop=1.0, num=size)
    x, y = np.meshgrid(coords, -coords)
    image = np.zeros((size, size))
    for amplitude, a, b, x0, y0, phi in _SHEPP_LOGAN_ELLIPSES:
        angle = np.deg2rad(phi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_shift, y_shift = x - x0, y - y0
        # Rotate into the ellipse's own frame before testing containment.
        x_rot = x_shift * cos_a + y_shift * sin_a
        y_rot = -x_shift * sin_a + y_shift * cos_a
        inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1.0
        image[inside] += amplitude
    return image
