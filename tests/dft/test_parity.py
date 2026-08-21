"""Tests for ``harmonics``/``AngularParity`` and the even-``N2`` Nyquist caveat.

An even angular sample count is just as valid as an odd one, but it has one
honest caveat -- the Nyquist harmonic ``-n_angular // 2``
has no ``+n_angular // 2`` partner in the harmonic range, so the conjugate
symmetry a real-valued signal's DFT normally has is one-sided there. This is
purely a property of ``harmonics``' derived range and ``angular_dft``'s
output, checked here so a regression (e.g. someone "fixing" the range to be
symmetric) would be caught immediately rather than waiting for the full PFT
pipeline's dB-error oracle.
"""

import numpy as np
import pytest

from pypft.dft import AngularParity, angular_dft, angular_parity, harmonics


@pytest.mark.parametrize(
    "n_angular,expected", [(15, AngularParity.ODD), (16, AngularParity.EVEN)]
)
def test_angular_parity_classifies_correctly(n_angular, expected):
    """``angular_parity`` reports the sample count's parity, not a judgment."""
    assert angular_parity(n_angular) == expected


@pytest.mark.parametrize("n_angular", [15, 16])
def test_harmonics_has_length_n_angular_and_holds_index_minus_half_size(n_angular):
    """``harmonics`` matches ``pypft.axes``'s centering convention exactly."""
    harms = harmonics(n_angular)
    assert harms.shape == (n_angular,)
    assert harms[0] == -(n_angular // 2)
    assert harms[-1] == n_angular - n_angular // 2 - 1


def test_odd_n_angular_harmonic_range_is_symmetric():
    """Odd sizes give every harmonic a ``-n``/``+n`` partner."""
    harms = set(int(n) for n in harmonics(15))
    assert harms == {-n for n in harms}


def test_even_n_angular_nyquist_harmonic_has_no_positive_partner():
    """Even sizes leave ``-n_angular // 2`` without a ``+n_angular // 2`` partner."""
    harms = set(int(n) for n in harmonics(16))
    nyquist = -8
    assert nyquist in harms
    assert -nyquist not in harms


def test_conjugate_symmetry_holds_except_at_the_even_nyquist_harmonic(implementation):
    """A real signal's DFT is conjugate-symmetric, except the unpaired Nyquist bin.

    Checked at both parities: for the odd case every harmonic has a partner
    and the check is exhaustive; for the even case the loop below skips
    exactly the one harmonic ``harmonics`` itself reports as unpaired, so the
    asymmetry stays visible rather than silently narrowing the test.
    """
    for n_angular in (15, 16):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(n_angular)  # real-valued, no imaginary part
        harms = harmonics(n_angular)
        harm_to_index = {int(n): i for i, n in enumerate(harms)}

        result = angular_dft(x, implementation)

        skipped_unpaired = 0
        for n, i in harm_to_index.items():
            if n == 0:
                continue
            j = harm_to_index.get(-n)
            if j is None:
                skipped_unpaired += 1
                continue
            np.testing.assert_allclose(
                result[i], np.conj(result[j]), rtol=1e-10, atol=1e-10
            )

        expected_unpaired = 1 if n_angular % 2 == 0 else 0
        assert skipped_unpaired == expected_unpaired
