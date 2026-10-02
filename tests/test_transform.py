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

from pypft.axes import PolarAxis
from pypft.dft import angular_dft, inverse_angular_dft
from pypft.grid import PolarGrid, sample_cartesian
from pypft.transform import (
    Direction,
    PFTImplementation,
    forward_pft,
    inverse_pft,
    scaled_hankel,
)

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
    grid = PolarGrid(n_radial=n_radial, n_angular=n_angular, R=_R)
    f = gaussian_f(grid.r.T)  # grid.r is (n_angular, n_radial); transpose to match
    F = forward_pft(f=f, grid=grid)
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
    avg_db, max_db = _forward_error_db(n_angular=n_angular, n_radial=n_radial)
    assert avg_db == pytest.approx(expected_avg, abs=0.15)
    assert max_db == pytest.approx(expected_max, abs=0.15)


def test_forward_pft_meets_the_accuracy_gate_at_the_papers_own_parameters():
    """The phase's own acceptance gate: ``E_avg < -60`` dB at N2=15, N1=383."""
    avg_db, _ = _forward_error_db(n_angular=15, n_radial=382)
    assert avg_db < -60.0


def test_forward_pft_meets_the_accuracy_gate_for_an_even_n_angular():
    """The same gate holds for an even ``n_angular`` (N2=16)."""
    avg_db, _ = _forward_error_db(n_angular=16, n_radial=382)
    assert avg_db < -60.0


def test_forward_pft_error_interleaves_between_odd_neighbours():
    """N2=16's error sits strictly between N2=15's and N2=17's, at N1=383.

    Machine-checks the even-N2 Nyquist caveat through the *full* PFT
    pipeline, complementing ``tests/dft/test_oracle.py``'s own check of the
    angular DFT composed directly with the DHT (without ``PolarGrid``/
    ``pypft.transform``): if a future change mishandled the unpaired
    Nyquist harmonic, N2=16 would jump out of this interleaved ordering.
    """
    avg_15, _ = _forward_error_db(n_angular=15, n_radial=382)
    avg_16, _ = _forward_error_db(n_angular=16, n_radial=382)
    avg_17, _ = _forward_error_db(n_angular=17, n_radial=382)
    assert avg_15 < avg_16 < avg_17


def test_inverse_pft_matches_the_papers_own_measured_error():
    """``inverse_pft`` reproduces Yao & Baddour Part II's own inverse figures."""
    grid = PolarGrid(n_radial=382, n_angular=15, R=_R)
    F = gaussian_F(grid.rho.T)
    f = inverse_pft(F=F, grid=grid)
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
    grid = PolarGrid(n_radial=64, n_angular=17, R=50.0)
    phantom = shepp_logan_phantom(128)
    sampled = sample_cartesian(
        image=phantom, grid=grid
    ).T  # (angular,radial)->(radial,angular)
    reconstructed = inverse_pft(F=forward_pft(f=sampled, grid=grid), grid=grid)
    np.testing.assert_allclose(reconstructed, sampled, rtol=1e-6, atol=1e-6)


def test_forward_pft_rejects_a_shape_mismatched_with_the_grid():
    """``forward_pft`` validates ``f``'s shape against the grid up front."""
    grid = PolarGrid(n_radial=382, n_angular=15, R=_R)
    wrong = np.zeros((382, 16))
    with pytest.raises(ValueError):
        forward_pft(f=wrong, grid=grid)


def test_scaled_hankel_rejects_a_non_direction_member():
    """``scaled_hankel`` type-validates its ``direction`` argument."""
    grid = PolarGrid(n_radial=382, n_angular=15, R=_R)
    values = np.zeros((382, 15))
    bad_direction: Direction = "forward"  # type: ignore[assignment]
    with pytest.raises(TypeError):
        scaled_hankel(
            values=values, grid=grid, direction=bad_direction, axis=0, angular_axis=1
        )


def test_scaled_hankel_agrees_between_axis_placements():
    """Transposing ``values`` and ``axis`` together leaves the result unchanged."""
    grid = PolarGrid(n_radial=382, n_angular=15, R=_R)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((382, 15)) + 1j * rng.standard_normal((382, 15))

    radial_first = scaled_hankel(
        values=values, grid=grid, direction=Direction.FORWARD, axis=0, angular_axis=1
    )
    angular_first = scaled_hankel(
        values=values.T, grid=grid, direction=Direction.FORWARD, axis=1, angular_axis=0
    )
    np.testing.assert_allclose(radial_first, angular_first.T, rtol=1e-12, atol=1e-12)


# ======================================================================================
# Batches of samples
# ======================================================================================

_BATCH_GRID = PolarGrid(n_radial=40, n_angular=15, R=_R)
_BATCH_SIZE = 4


def _random_batch(rng: np.random.Generator, batch: int) -> np.ndarray:
    """A ``(n_radial, n_angular, batch)`` array of independent random signals."""
    shape = (_BATCH_GRID.n_radial, _BATCH_GRID.n_angular, batch)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_forward_pft_of_a_batch_matches_looping_over_its_samples():
    """A batched ``forward_pft`` call matches transforming each slice on its own."""
    rng = np.random.default_rng(10)
    f = _random_batch(rng, _BATCH_SIZE)

    batched = forward_pft(f=f, grid=_BATCH_GRID)

    looped = np.stack(
        [forward_pft(f=f[..., b], grid=_BATCH_GRID) for b in range(_BATCH_SIZE)],
        axis=-1,
    )
    np.testing.assert_allclose(batched, looped, rtol=1e-10, atol=1e-10)


def test_inverse_pft_of_a_batch_matches_looping_over_its_samples():
    """A batched ``inverse_pft`` call matches transforming each slice on its own."""
    rng = np.random.default_rng(11)
    F = _random_batch(rng, _BATCH_SIZE)

    batched = inverse_pft(F=F, grid=_BATCH_GRID)

    looped = np.stack(
        [inverse_pft(F=F[..., b], grid=_BATCH_GRID) for b in range(_BATCH_SIZE)],
        axis=-1,
    )
    np.testing.assert_allclose(batched, looped, rtol=1e-10, atol=1e-10)


def test_forward_pft_batch_of_one_matches_the_single_sample_result():
    """A length-1 batch axis reproduces the single-sample result exactly."""
    rng = np.random.default_rng(12)
    f = rng.standard_normal((_BATCH_GRID.n_radial, _BATCH_GRID.n_angular))

    plain = forward_pft(f=f, grid=_BATCH_GRID)
    batched = forward_pft(f=f[..., np.newaxis], grid=_BATCH_GRID)

    np.testing.assert_allclose(batched[..., 0], plain, rtol=1e-12, atol=1e-12)


def test_forward_pft_rejects_a_batch_axis_on_a_single_sample():
    """A single sample has no batch axis to place -- passing one is a caller error."""
    f = np.zeros((_BATCH_GRID.n_radial, _BATCH_GRID.n_angular))
    with pytest.raises(ValueError):
        forward_pft(f=f, grid=_BATCH_GRID, batch_axis=0)


def test_forward_pft_rejects_a_batch_axis_that_is_not_last():
    """PyPFT's own layout always places the batch axis after the sample axes."""
    f = np.zeros((_BATCH_GRID.n_radial, _BATCH_GRID.n_angular, _BATCH_SIZE))
    with pytest.raises(ValueError, match="after the sample axes"):
        forward_pft(f=f, grid=_BATCH_GRID, batch_axis=0)


def test_a_batch_round_trips():
    """``inverse_pft(forward_pft(f))`` returns every sample of a batch."""
    rng = np.random.default_rng(15)
    f = _random_batch(rng, _BATCH_SIZE)

    round_tripped = inverse_pft(F=forward_pft(f=f, grid=_BATCH_GRID), grid=_BATCH_GRID)

    # The DHT's own self-inverse residual grows with order (tests/dht/tolerance.py);
    # the highest harmonic order here is n_angular // 2 == 7.
    np.testing.assert_allclose(round_tripped, f, rtol=1e-4, atol=1e-4)


#: Neither a single sample nor a batch: a bare radial line, and a batch with one
#: extra axis.
_WRONG_RANK_SHAPES = [
    (_BATCH_GRID.n_radial,),
    (_BATCH_GRID.n_radial, _BATCH_GRID.n_angular, _BATCH_SIZE, 2),
]

#: What every polar entry point says when given any other rank.
_WRONG_RANK_MESSAGE = (
    r"a single sample \(radial, angular\) or a batch \(radial, angular, batch\)"
)


@pytest.mark.parametrize(argnames="shape", argvalues=_WRONG_RANK_SHAPES)
@pytest.mark.parametrize(
    argnames=("transform", "argument"),
    argvalues=[(forward_pft, "f"), (inverse_pft, "F")],
)
def test_any_other_rank_raises_naming_sample_and_batch(transform, argument, shape):
    """A 4-D (or 1-D) array is neither a sample nor a batch, and says so."""
    with pytest.raises(ValueError, match=_WRONG_RANK_MESSAGE):
        transform(**{argument: np.zeros(shape=shape)}, grid=_BATCH_GRID)


@pytest.mark.parametrize("implementation", list(PFTImplementation))
def test_scaled_hankel_implementations_agree_on_a_single_sample(implementation):
    """``HARMONIC_LOOP`` and ``STACKED_KERNEL`` compute the exact same result."""
    rng = np.random.default_rng(13)
    values = rng.standard_normal(
        (_BATCH_GRID.n_radial, _BATCH_GRID.n_angular)
    ) + 1j * rng.standard_normal((_BATCH_GRID.n_radial, _BATCH_GRID.n_angular))

    result = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=Direction.FORWARD,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
        implementation=implementation,
    )
    reference = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=Direction.FORWARD,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
        implementation=PFTImplementation.HARMONIC_LOOP,
    )
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("implementation", list(PFTImplementation))
def test_scaled_hankel_implementations_agree_on_a_batch(implementation):
    """The two implementations also agree on a batch of samples."""
    rng = np.random.default_rng(14)
    values = _random_batch(rng, _BATCH_SIZE)

    result = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=Direction.INVERSE,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
        implementation=implementation,
    )
    reference = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=Direction.INVERSE,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
        implementation=PFTImplementation.HARMONIC_LOOP,
    )
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)


# ======================================================================================
# Axis flexibility of the steps under forward_pft/inverse_pft
# ======================================================================================

#: Where the radial and angular axes sit in a 4-D layout unlike the polar one: the
#: harmonic axis last, the radial axis second, two other axes around them.
_FLEXIBLE_RADIAL_AXIS = 1
_FLEXIBLE_ANGULAR_AXIS = 3
_FLEXIBLE_EXTRA_SHAPE = (3, 2)


def _flexible_layout(values: np.ndarray) -> np.ndarray:
    """Lay a ``(radial, angular, 3, 2)`` array out as ``(3, radial, 2, angular)``."""
    return np.moveaxis(
        a=values,
        source=(PolarAxis.RADIAL, PolarAxis.ANGULAR, 2, 3),
        destination=(_FLEXIBLE_RADIAL_AXIS, _FLEXIBLE_ANGULAR_AXIS, 0, 2),
    )


def _random_flexible_values(rng: np.random.Generator) -> np.ndarray:
    """A random complex ``(radial, angular, 3, 2)`` array on ``_BATCH_GRID``."""
    shape = (_BATCH_GRID.n_radial, _BATCH_GRID.n_angular) + _FLEXIBLE_EXTRA_SHAPE
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("implementation", list(PFTImplementation))
def test_scaled_hankel_gives_the_same_result_in_any_axis_layout(
    implementation, direction
):
    """Moving the radial/harmonic axes elsewhere in an N-D array moves the result."""
    rng = np.random.default_rng(16)
    values = _random_flexible_values(rng)

    polar = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=direction,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
        implementation=implementation,
    )
    flexible = scaled_hankel(
        values=_flexible_layout(values=values),
        grid=_BATCH_GRID,
        direction=direction,
        axis=_FLEXIBLE_RADIAL_AXIS,
        angular_axis=_FLEXIBLE_ANGULAR_AXIS,
        implementation=implementation,
    )
    np.testing.assert_allclose(
        flexible, _flexible_layout(values=polar), rtol=1e-12, atol=1e-12
    )


def test_scaled_hankel_on_n_d_input_matches_looping_over_the_other_axes():
    """Every axis but the radial and harmonic ones is carried along unchanged."""
    rng = np.random.default_rng(17)
    values = _random_flexible_values(rng)

    batched = scaled_hankel(
        values=values,
        grid=_BATCH_GRID,
        direction=Direction.FORWARD,
        axis=PolarAxis.RADIAL,
        angular_axis=PolarAxis.ANGULAR,
    )
    for i, j in np.ndindex(*_FLEXIBLE_EXTRA_SHAPE):
        single = scaled_hankel(
            values=values[:, :, i, j],
            grid=_BATCH_GRID,
            direction=Direction.FORWARD,
            axis=PolarAxis.RADIAL,
            angular_axis=PolarAxis.ANGULAR,
        )
        np.testing.assert_allclose(batched[:, :, i, j], single, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    argnames=("transform", "argument"),
    argvalues=[(angular_dft, "x"), (inverse_angular_dft, "X")],
)
def test_the_angular_dft_gives_the_same_result_in_any_axis_layout(transform, argument):
    """The angular DFT/IDFT follow their angular axis wherever it sits."""
    rng = np.random.default_rng(18)
    values = _random_flexible_values(rng)

    polar = transform(**{argument: values}, axis=PolarAxis.ANGULAR)
    flexible = transform(
        **{argument: _flexible_layout(values=values)}, axis=_FLEXIBLE_ANGULAR_AXIS
    )

    np.testing.assert_allclose(
        flexible, _flexible_layout(values=polar), rtol=1e-12, atol=1e-12
    )
