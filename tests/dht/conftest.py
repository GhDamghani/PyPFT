"""Fixtures for the DHT test suite.

Radial profiles are extracted from ``tests/samples/lena.tif`` via OpenCV's polar
warp. These are used only for round-trip (``inverse(forward(f)) ~= f``) and
cross-implementation agreement checks: there is no closed-form Hankel transform
of an arbitrary photographic radial profile, so they are never compared against
a known analytical answer.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from pypft.dht import DHTImplementation

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "samples" / "lena.tif"

#: Length of the vectors transformed in tests, chosen comfortably above the
#: N > 30 threshold above which baddour2019.md reports ~1e-7 orthogonality
#: error (Eq. 37 discussion).
SIGNAL_SIZE = 64

#: Orders exercised across the parametrized test suite: 0 (the common case,
#: with a closed-form self-reciprocal Gaussian check), a few small arbitrary
#: nonzero orders, and high orders (16, 32, 64) that specifically exercise
#: the order-dependent kernel degradation that let ``RecurrenceBesselDHT``
#: diverge unnoticed. Order-sensitive assertions must use
#: ``tests.dht.tolerance.dht_tolerance`` instead of the flat RTOL/ATOL below
#: once orders this high are involved.
DHT_ORDERS = (0, 1, 4, 16, 32, 64)

ALL_IMPLEMENTATIONS = tuple(DHTImplementation)

#: Tolerance for SIGNAL_SIZE = 64 (> 30), matching baddour2019.md's own reported
#: ~1e-7 orthogonality error in that regime (Eq. 37 discussion), with a ~10x
#: margin observed empirically to hold up through order n=4. Only valid for
#: order-insensitive assertions -- see ``tests.dht.tolerance`` for orders
#: above ~12, where this flat bound is measurably too tight.
RTOL = 1e-6
ATOL = 1e-6


@pytest.fixture(scope="session")
def lena_image() -> np.ndarray:
    """The lena test image as a 2-D ``float64`` grayscale array."""
    image = cv2.imread(filename=str(SAMPLES_PATH), flags=cv2.IMREAD_GRAYSCALE)
    assert image is not None, f"failed to read {SAMPLES_PATH}"
    return image.astype(np.float64)


@pytest.fixture(scope="session")
def lena_radial_profiles(lena_image: np.ndarray) -> list[np.ndarray]:
    """A handful of length-``SIGNAL_SIZE`` radial profiles cut from lena.

    Uses ``cv2.warpPolar`` to resample the image on an (angle, radius) grid,
    then returns a few individual angle cuts plus the angular average, each a
    1-D radial profile ``f(r)`` mirroring PyPFT's own ``(r, theta)`` domain.

    :param lena_image: The source grayscale image.
    :type lena_image: np.ndarray
    :returns: A list of length-``SIGNAL_SIZE`` 1-D radial profiles.
    :rtype: list[np.ndarray]

    """
    height, width = lena_image.shape
    center = (width / 2, height / 2)
    max_radius = min(height, width) / 2
    num_angles = 16
    polar = cv2.warpPolar(
        src=lena_image,
        dsize=(SIGNAL_SIZE, num_angles),
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR,
    )
    profiles = [polar[i, :] for i in range(0, num_angles, 4)]
    profiles.append(polar.mean(axis=0))
    return profiles


@pytest.fixture(params=ALL_IMPLEMENTATIONS, ids=lambda impl: impl.name)
def implementation(request: pytest.FixtureRequest) -> DHTImplementation:
    """Each ``DHTImplementation`` strategy, in turn."""
    return request.param


@pytest.fixture(params=DHT_ORDERS, ids=lambda n: f"n={n}")
def order(request: pytest.FixtureRequest) -> int:
    """Each order exercised by the test suite, in turn."""
    return request.param
