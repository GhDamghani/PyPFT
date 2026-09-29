"""Tests for the typed domain objects (``pypft.domains``).

``BaseSignal``'s four subclasses are a thin, typed shell over the already-verified
``pypft.transform``/``pypft.dft`` numerics -- these tests check the shell itself:
that each legal edge round-trips, that ``to`` composes the full chain to match
``forward_pft``/``inverse_pft`` exactly, that an illegal hand-written chain is both a
``pyright`` error (the ``# type: ignore`` on the call below) and a runtime
``AttributeError``, and that the dynamic ``to`` still validates its own argument.
"""

import numpy as np
import pytest

from pypft.domains import (
    _CHAIN,
    _STEP_BACKWARD,
    _STEP_TOWARD,
    BaseSignal,
    Domain,
    PolarFrequencyHarmonicSignal,
    PolarFrequencySignal,
    PolarSpatialHarmonicSignal,
    PolarSpatialSignal,
)
from pypft.grid import PolarGrid
from pypft.transform import forward_pft, inverse_pft

_R = 40.0
_N_RADIAL = 32
_N_ANGULAR = 15


def _random_values(rng: np.random.Generator) -> np.ndarray:
    """Build a random complex ``(n_radial, n_angular)`` array for the tests' grid."""
    shape = (_N_RADIAL, _N_ANGULAR)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# ======================================================================================
# Construction and validation
# ======================================================================================


def test_each_subclass_is_tagged_with_its_own_domain():
    """Every ``BaseSignal`` subclass fixes ``domain`` to its own ``Domain`` member."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex)
    assert PolarSpatialSignal(values=values, grid=grid).domain is Domain.POLAR_SPATIAL
    assert (
        PolarSpatialHarmonicSignal(values=values, grid=grid).domain
        is Domain.POLAR_SPATIAL_HARMONIC
    )
    assert (
        PolarFrequencyHarmonicSignal(values=values, grid=grid).domain
        is Domain.POLAR_FREQUENCY_HARMONIC
    )
    assert (
        PolarFrequencySignal(values=values, grid=grid).domain is Domain.POLAR_FREQUENCY
    )


def test_construction_rejects_a_shape_mismatched_with_the_grid():
    """``BaseSignal.__post_init__`` validates ``values``'s shape against ``grid``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    wrong = np.zeros((_N_RADIAL, _N_ANGULAR + 1), dtype=complex)
    with pytest.raises(ValueError):
        PolarSpatialSignal(values=wrong, grid=grid)


def test_construction_rejects_a_non_ndarray_values_argument():
    """``BaseSignal.__post_init__`` type-validates ``values``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    with pytest.raises(TypeError):
        PolarSpatialSignal(values=[[0.0]], grid=grid)  # type: ignore[arg-type]


def test_construction_accepts_a_3d_batch():
    """``values`` may add a trailing batch axis on top of the plain 2-D case."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR, 3), dtype=complex)
    signal = PolarSpatialSignal(values=values, grid=grid)
    assert signal.values.shape == (_N_RADIAL, _N_ANGULAR, 3)


def test_construction_rejects_a_batch_axis_on_2d_values():
    """A 2-D ``values`` has no batch axis to place -- passing one is a caller error."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex)
    with pytest.raises(ValueError):
        PolarSpatialSignal(values=values, grid=grid, batch_axis=0)


def test_construction_rejects_a_batch_axis_that_is_not_last():
    """PyPFT's own layout always places the batch axis last (Axis.BATCH)."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR, 3), dtype=complex)
    with pytest.raises(ValueError):
        PolarSpatialSignal(values=values, grid=grid, batch_axis=0)


# ======================================================================================
# The chain and its step-method names
# ======================================================================================

#: The PFT's domain chain, spatial to frequency, spelled out by name.
_EXPECTED_CHAIN_NAMES = [
    "POLAR_SPATIAL",
    "POLAR_SPATIAL_HARMONIC",
    "POLAR_FREQUENCY_HARMONIC",
    "POLAR_FREQUENCY",
]


def test_the_chain_is_ordered_spatial_to_frequency():
    """``_CHAIN`` lists every ``Domain`` member in the PFT's own step order."""
    assert [domain.name for domain in _CHAIN] == _EXPECTED_CHAIN_NAMES


@pytest.mark.parametrize("edge", range(len(_EXPECTED_CHAIN_NAMES) - 1))
def test_every_step_method_is_named_after_its_destination_domain(edge: int):
    """Each step method is ``to_<destination domain>``, in both directions."""
    assert _STEP_TOWARD[edge] == f"to_{_CHAIN[edge + 1].name.lower()}"
    assert _STEP_BACKWARD[edge] == f"to_{_CHAIN[edge].name.lower()}"


# ======================================================================================
# Every legal edge round-trips
# ======================================================================================


def test_the_spatial_angular_edge_round_trips():
    """The spatial domain's angular DFT/IDFT edge is an inverse pair."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(0)
    original = PolarSpatialSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_polar_spatial_harmonic().to_polar_spatial()

    np.testing.assert_allclose(round_tripped.values, original.values, atol=1e-10)


def test_the_scaled_hankel_edge_round_trips():
    """The scaled Hankel transform edge, spatial to frequency, is an inverse pair."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(1)
    original = PolarSpatialHarmonicSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_polar_frequency_harmonic().to_polar_spatial_harmonic()

    # rtol matches the DHT's own order-dependent residual (tests/dht/tolerance.py):
    # the highest harmonic order here is n_angular // 2 == 7, not order 0.
    np.testing.assert_allclose(round_tripped.values, original.values, rtol=1e-4)


def test_the_frequency_angular_edge_round_trips():
    """The frequency domain's own angular DFT/IDFT edge is likewise an inverse pair."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(2)
    original = PolarFrequencyHarmonicSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_polar_frequency().to_polar_frequency_harmonic()

    np.testing.assert_allclose(round_tripped.values, original.values, atol=1e-10)


# ======================================================================================
# ``to`` composes the full chain
# ======================================================================================


def test_to_composes_the_full_forward_chain_matching_forward_pft():
    """Walking ``POLAR_SPATIAL`` to ``POLAR_FREQUENCY`` matches ``forward_pft``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    f = np.exp(-(grid.r.T**2))
    signal = PolarSpatialSignal(values=f, grid=grid)

    walked = signal.to(Domain.POLAR_FREQUENCY)

    expected = forward_pft(f=f, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.domain is Domain.POLAR_FREQUENCY


def test_to_composes_the_full_backward_chain_matching_inverse_pft():
    """Walking ``POLAR_FREQUENCY`` to ``POLAR_SPATIAL`` matches ``inverse_pft``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    F = np.pi * np.exp(-(grid.rho.T**2) / 4.0)
    signal = PolarFrequencySignal(values=F, grid=grid)

    walked = signal.to(Domain.POLAR_SPATIAL)

    expected = inverse_pft(F=F, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.domain is Domain.POLAR_SPATIAL


def test_to_composes_the_full_forward_chain_for_a_3d_batch():
    """``to`` also composes correctly with a trailing batch axis present."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    f = np.broadcast_to(
        array=np.exp(-(grid.r.T**2))[..., np.newaxis], shape=(_N_RADIAL, _N_ANGULAR, 3)
    ).copy()
    signal = PolarSpatialSignal(values=f, grid=grid)

    walked = signal.to(Domain.POLAR_FREQUENCY)

    expected = forward_pft(f=f, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.batch_axis == signal.batch_axis


def test_to_a_signals_own_domain_is_a_no_op():
    """Walking to the domain a signal is already in returns it unchanged."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(3)
    signal = PolarSpatialHarmonicSignal(values=_random_values(rng), grid=grid)

    walked = signal.to(Domain.POLAR_SPATIAL_HARMONIC)

    np.testing.assert_array_equal(walked.values, signal.values)


def test_to_rejects_a_non_domain_type():
    """The dynamic ``to`` type-validates its ``domain`` argument."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal = PolarSpatialSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(TypeError):
        signal.to("POLAR_FREQUENCY")  # type: ignore[arg-type]


def test_to_rejects_an_enum_member_of_the_wrong_type():
    """The dynamic ``to`` rejects an enum member that isn't a ``Domain``."""
    from pypft.transform import Direction

    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal = PolarSpatialSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(ValueError):
        signal.to(Direction.FORWARD)  # type: ignore[arg-type]


# ======================================================================================
# Illegal edges are pyright errors on hand-written chains
# ======================================================================================


def test_a_hand_written_illegal_edge_is_not_a_valid_attribute():
    """``PolarSpatialSignal`` cannot skip the angular DFT to reach the frequency domain.

    ``to_polar_frequency_harmonic`` is a non-adjacent edge from ``POLAR_SPATIAL``. The
    ``# type: ignore[attr-defined]`` below marks exactly what ``pyright`` would
    otherwise reject on this hand-written chain: only the neighbouring subclass
    (``PolarSpatialHarmonicSignal``) defines ``to_polar_frequency_harmonic``. Removing
    the comment makes the quality gate's own ``pyright`` step fail.
    """
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal: BaseSignal = PolarSpatialSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(AttributeError):
        signal.to_polar_frequency_harmonic()  # type: ignore[attr-defined]
