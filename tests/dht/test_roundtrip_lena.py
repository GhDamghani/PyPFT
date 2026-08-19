"""Round-trip and cross-implementation consistency checks using lena radial profiles.

There is no closed-form Hankel transform of an arbitrary photographic radial
profile, so these checks never compare against a known analytical answer -- only
against themselves (round trip) and against each other (cross-implementation
agreement), per the plan's "lena fixture role" decision.
"""

import numpy as np

from pypft.dht import DHTImplementation, hankel_transform, inverse_hankel_transform

from .conftest import ALL_IMPLEMENTATIONS, ATOL, RTOL

_R = 1.0


def test_forward_inverse_round_trip_on_lena_profiles(
    implementation, order, lena_radial_profiles
):
    """``inverse(forward(f)) ~= f`` for every lena-derived radial profile."""
    for f in lena_radial_profiles:
        F = hankel_transform(f, order, _R, implementation)
        reconstructed = inverse_hankel_transform(F, order, _R, implementation)
        np.testing.assert_allclose(reconstructed, f, rtol=RTOL, atol=ATOL)


def test_implementations_agree_on_lena_profiles(order, lena_radial_profiles):
    """Every implementation produces the same forward transform on real data."""
    reference = DHTImplementation.NAIVE
    for f in lena_radial_profiles:
        expected = hankel_transform(f, order, _R, reference)
        for implementation in ALL_IMPLEMENTATIONS:
            if implementation is reference:
                continue
            actual = hankel_transform(f, order, _R, implementation)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"{implementation} disagrees with {reference}",
            )
