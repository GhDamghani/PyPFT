"""The angular DFT composes with the existing DHT to reproduce the paper's PFT.

Assembles a verified forward-PFT prototype (Yao & Baddour Part II, its
supplementary MATLAB appendix) using ``angular_dft``/``inverse_angular_dft``
in place of the prototype's own raw ``fftshift(fft(ifftshift(...)))`` calls
-- proving this subsystem's centering and harmonic-range derivation compose
correctly with ``pypft.dht``, without introducing ``PolarGrid``/
``transform.py``. This is a stronger check than testing ``pypft.dft`` in
isolation: a wrong harmonic range, a mis-centered output, or an off-by-one in
the Nyquist handling would all show up as a wrong dB error here, not just a
wrong index.

The grid construction, per-harmonic sign/scale factors, and analytic oracle
all mirror that prototype exactly; only the two
``fftshift``/``fft``/``ifftshift`` triples are replaced by calls into this
module's public API.
"""

import numpy as np
import pytest
from scipy.special import jn_zeros

from pypft.dft import angular_dft, harmonics, inverse_angular_dft
from pypft.dht._cached import CachedBesselDHT

_R = 40.0
_N1 = 383
_SIZE = _N1 - 1

#: ``(n_angular, expected_E_avg_dB, expected_E_max_dB)``, measured directly
#: from this exact computation. Fully deterministic (no RNG anywhere in the
#: chain), so the tolerance in the test below only has to absorb
#: floating-point noise between FFT backends.
_MEASURED = [
    (15, -63.80, -8.38),
    (16, -62.39, -7.72),
    (17, -61.92, -7.15),
]


def _forward_oracle_error_db(n_angular, implementation):
    """Forward-Gaussian dB error of the DFT-DHT-IDFT chain at ``n_angular``.

    :param n_angular: The number of angular samples (``N2``).
    :type n_angular: int
    :param implementation: The ``DFTImplementation`` to route ``angular_dft``/
        ``inverse_angular_dft`` through.
    :type implementation: pypft.dft.DFTImplementation
    :returns: The mean and max dB error against the analytic oracle.
    :rtype: tuple[float, float]

    """
    harms = harmonics(n_angular)
    r = np.empty((n_angular, _SIZE))
    rho = np.empty((n_angular, _SIZE))
    for i, n in enumerate(harms):
        zeros = jn_zeros(abs(int(n)), _SIZE + 1)  # order-dependent radial grid
        r[i, :] = zeros[:-1] / zeros[-1] * _R
        rho[i, :] = zeros[:-1] / _R

    f = np.exp(-(r**2))
    fnk = angular_dft(f, implementation, axis=0)
    Fnl = np.empty_like(fnk)
    for i, n in enumerate(harms):
        n = int(n)
        sign = (-1.0) ** abs(n) if n < 0 else 1.0  # negative-order relation
        dht = CachedBesselDHT.forward(fnk[i, :], abs(n), _R)
        Fnl[i, :] = sign * dht * (2 * np.pi) * (1j ** (-n))  # per-harmonic scale
    F = inverse_angular_dft(Fnl, implementation, axis=0)

    expected = np.pi * np.exp(-(rho**2) / 4)  # the analytic Hankel-transform oracle
    err_db = 20 * np.log10(np.abs(expected - F) / np.max(np.abs(F)))
    return float(err_db.mean()), float(err_db.max())


@pytest.mark.parametrize("n_angular,expected_avg,expected_max", _MEASURED)
def test_angular_dft_composes_with_dht_to_match_the_published_pft_error(
    n_angular, expected_avg, expected_max, implementation
):
    """The DFT+DHT+IDFT chain matches Yao & Baddour Part II's own figures."""
    avg_db, max_db = _forward_oracle_error_db(n_angular, implementation)
    assert avg_db == pytest.approx(expected_avg, abs=0.1)
    assert max_db == pytest.approx(expected_max, abs=0.1)


def test_even_n_angular_error_interleaves_between_its_odd_neighbours(implementation):
    """N2=16's error sits strictly between N2=15's and N2=17's.

    The machine-checked form of the even-``N2`` Nyquist caveat: if a future
    change mishandled the unpaired Nyquist harmonic, N2=16 would jump out of
    this interleaved ordering rather than smoothly continuing the trend its
    odd neighbours set.
    """
    avg_15, _ = _forward_oracle_error_db(15, implementation)
    avg_16, _ = _forward_oracle_error_db(16, implementation)
    avg_17, _ = _forward_oracle_error_db(17, implementation)

    assert avg_15 < avg_16 < avg_17
