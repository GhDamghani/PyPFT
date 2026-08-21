"""Tests for the full PFT/IPFT pipeline (``pypft.transform``).

Ports Yao & Baddour's own verified MATLAB appendix (PeerJ CS Part II,
Appendix A-5/A-6) through ``pypft.grid.PolarGrid`` and
``pypft.transform.forward_pft``/``inverse_pft``, reproducing that paper's own
published dB-error figures bit-for-bit (up to floating-point noise) and its
own accuracy trade-off across grid sizes -- mirrored here as several
``(n_angular, n_radial)`` combinations rather than one flat threshold.
``E_max`` legitimately occurs at the grid's central gap and can even turn
positive at an inadequate grid size, so every gate below checks the
*average* dB error, never the max (PeerJ CS Part II).
"""

import numpy as np
import pytest

from pypft.grid import PolarGrid, sample_cartesian
from pypft.transform import Direction, forward_pft, inverse_pft, scaled_hankel

from .fixtures import gaussian_F, gaussian_f, shepp_logan_phantom

_R = 40.0

#: ``(n_angular, n_radial, expected_avg_dB, expected_max_dB)``, measured
#: directly from this exact forward-PFT computation (``n_radial`` here is
#: Baddour's "N1 - 1", so ``N1=383`` is ``n_radial=382``). The first three
#: rows are Yao & Baddour Part II's own published figures (``N1=383``); the
#: rest extend across ``n_radial`` to show the N1-must-grow-with-N2
#: trade-off. The paper's own ``N1=1535`` rows are omitted here to keep the
#: routine suite fast -- each extra row costs a full per-harmonic kernel
#: build up to that row's highest order.
_FORWARD_MEASURED = [
    (15, 382, -63.80, -8.38),
    (16, 382, -62.39, -7.72),
    (17, 382, -61.92, -7.15),
    (15, 766, -88.37, -14.95),
    (32, 382, -52.05, -0.65),
    (32, 766, -76.76, -7.72),
    (64, 382, -41.40, 6.93),
    (64, 766, -65.25, -0.66),
]


def _forward_error_db(n_angular: int, n_radial: int) -> tuple[float, float]:
    """Compute the forward-PFT dB error of the Gaussian oracle.

    :param n_angular: The number of angular samples.
    :type n_angular: int
    :param n_radial: The number of radial samples.
    :type n_radial: int
    :returns: The mean and max dB error against the analytic oracle.
    :rtype: tuple[float, float]

    """
    grid = PolarGrid(n_radial, n_angular, _R)
    f = gaussian_f(grid.r.T)  # grid.r is (n_angular, n_radial); transpose to match
    F = forward_pft(f, grid)
    expected = gaussian_F(grid.rho.T)
    err_db = 20 * np.log10(np.abs(expected - F) / np.max(np.abs(F)))
    return float(err_db.mean()), float(err_db.max())


@pytest.mark.parametrize(
    "n_angular,n_radial,expected_avg,expected_max", _FORWARD_MEASURED
)
def test_forward_pft_matches_measured_error_across_grid_sizes(
    n_angular, n_radial, expected_avg, expected_max
):
    """``forward_pft`` reproduces the measured average/max dB error at each size."""
    avg_db, max_db = _forward_error_db(n_angular, n_radial)
    assert avg_db == pytest.approx(expected_avg, abs=0.15)
    assert max_db == pytest.approx(expected_max, abs=0.15)


def test_forward_pft_meets_the_accuracy_gate_at_the_papers_own_parameters():
    """The phase's own acceptance gate: ``E_avg < -60`` dB at N2=15, N1=383."""
    avg_db, _ = _forward_error_db(15, 382)
    assert avg_db < -60.0


def test_forward_pft_meets_the_accuracy_gate_for_an_even_n_angular():
    """The same gate holds for an even ``n_angular`` (N2=16)."""
    avg_db, _ = _forward_error_db(16, 382)
    assert avg_db < -60.0


def test_forward_pft_error_interleaves_between_odd_neighbours():
    """N2=16's error sits strictly between N2=15's and N2=17's, at N1=383.

    Machine-checks the even-N2 Nyquist caveat through the *full* PFT
    pipeline, complementing ``tests/dft/test_oracle.py``'s own check of the
    angular DFT composed directly with the DHT (without ``PolarGrid``/
    ``pypft.transform``): if a future change mishandled the unpaired
    Nyquist harmonic, N2=16 would jump out of this interleaved ordering.
    """
    avg_15, _ = _forward_error_db(15, 382)
    avg_16, _ = _forward_error_db(16, 382)
    avg_17, _ = _forward_error_db(17, 382)
    assert avg_15 < avg_16 < avg_17


def test_inverse_pft_matches_the_papers_own_measured_error():
    """``inverse_pft`` reproduces Yao & Baddour Part II's own inverse figures."""
    grid = PolarGrid(382, 15, _R)
    F = gaussian_F(grid.rho.T)
    f = inverse_pft(F, grid)
    expected = gaussian_f(grid.r.T)
    err_db = 20 * np.log10(np.abs(expected - f) / np.max(np.abs(f)))
    assert float(err_db.mean()) == pytest.approx(-98.03, abs=0.1)
    assert float(err_db.max()) == pytest.approx(-12.26, abs=0.1)
    assert float(err_db.mean()) < -95.0


def test_round_trip_recovers_a_sampled_phantom_as_a_regression_only():
    """Round trip stays near machine precision -- a regression check only.

    Per PeerJ CS Part II (and this package's own angular-DFT measurements),
    the DHT's self-inverse kernel and the forward/inverse scale factors
    cancel exactly across a forward/inverse pair regardless of whether the
    forward transform is itself numerically accurate at these parameters --
    a round trip can never certify accuracy, only catch a regression in the
    composition. The Shepp-Logan phantom is used here specifically because
    it is *not* circularly symmetric, unlike the Gaussian oracle above.
    """
    grid = PolarGrid(64, 17, R=50.0)
    phantom = shepp_logan_phantom(128)
    sampled = sample_cartesian(phantom, grid).T  # (angular,radial)->(radial,angular)
    reconstructed = inverse_pft(forward_pft(sampled, grid), grid)
    np.testing.assert_allclose(reconstructed, sampled, rtol=1e-6, atol=1e-6)


def test_forward_pft_rejects_a_shape_mismatched_with_the_grid():
    """``forward_pft`` validates ``f``'s shape against the grid up front."""
    grid = PolarGrid(382, 15, _R)
    wrong = np.zeros((382, 16))
    with pytest.raises(ValueError):
        forward_pft(wrong, grid)


def test_scaled_hankel_rejects_a_non_direction_member():
    """``scaled_hankel`` type-validates its ``direction`` argument."""
    grid = PolarGrid(382, 15, _R)
    values = np.zeros((382, 15))
    bad_direction: Direction = "forward"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        scaled_hankel(values, grid, direction=bad_direction, axis=0)


def test_scaled_hankel_agrees_between_axis_placements():
    """Transposing ``values`` and ``axis`` together leaves the result unchanged."""
    grid = PolarGrid(382, 15, _R)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((382, 15)) + 1j * rng.standard_normal((382, 15))

    radial_first = scaled_hankel(values, grid, direction=Direction.FORWARD, axis=0)
    angular_first = scaled_hankel(values.T, grid, direction=Direction.FORWARD, axis=1)
    np.testing.assert_allclose(radial_first, angular_first.T, rtol=1e-12, atol=1e-12)
