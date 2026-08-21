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
    BaseSignal,
    Domain,
    FrequencyHarmonicSignal,
    FrequencyPolarSignal,
    SpaceHarmonicSignal,
    SpacePolarSignal,
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
    assert SpacePolarSignal(values=values, grid=grid).domain is Domain.SPACE_POLAR
    assert SpaceHarmonicSignal(values=values, grid=grid).domain is Domain.SPACE_HARMONIC
    assert (
        FrequencyHarmonicSignal(values=values, grid=grid).domain
        is Domain.FREQUENCY_HARMONIC
    )
    assert (
        FrequencyPolarSignal(values=values, grid=grid).domain is Domain.FREQUENCY_POLAR
    )


def test_construction_rejects_a_shape_mismatched_with_the_grid():
    """``BaseSignal.__post_init__`` validates ``values``'s shape against ``grid``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    wrong = np.zeros((_N_RADIAL, _N_ANGULAR + 1), dtype=complex)
    with pytest.raises(ValueError):
        SpacePolarSignal(values=wrong, grid=grid)


def test_construction_rejects_a_non_ndarray_values_argument():
    """``BaseSignal.__post_init__`` type-validates ``values``."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    with pytest.raises(TypeError):
        SpacePolarSignal(values=[[0.0]], grid=grid)  # type: ignore[arg-type]


def test_construction_accepts_a_3d_batch():
    """``values`` may add a trailing batch axis on top of the plain 2-D case."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR, 3), dtype=complex)
    signal = SpacePolarSignal(values=values, grid=grid)
    assert signal.values.shape == (_N_RADIAL, _N_ANGULAR, 3)


def test_construction_rejects_a_batch_axis_on_2d_values():
    """A 2-D ``values`` has no batch axis to place -- passing one is a caller error."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex)
    with pytest.raises(ValueError):
        SpacePolarSignal(values=values, grid=grid, batch_axis=0)


def test_construction_rejects_a_batch_axis_that_is_not_last():
    """PyPFT's own layout always places the batch axis last (Axis.BATCH)."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    values = np.zeros((_N_RADIAL, _N_ANGULAR, 3), dtype=complex)
    with pytest.raises(ValueError):
        SpacePolarSignal(values=values, grid=grid, batch_axis=0)


# ======================================================================================
# Every legal edge round-trips
# ======================================================================================


def test_the_polar_harmonic_edge_round_trips():
    """``to_harmonics``/``to_angles`` (the angular DFT/IDFT edge) are inverses."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(0)
    original = SpacePolarSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_harmonics().to_angles()

    np.testing.assert_allclose(round_tripped.values, original.values, atol=1e-10)


def test_the_space_frequency_edge_round_trips():
    """``to_frequency``/``to_space`` (the scaled Hankel transform edge) are inverses."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(1)
    original = SpaceHarmonicSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_frequency().to_space()

    # rtol matches the DHT's own order-dependent residual (tests/dht/tolerance.py):
    # the highest harmonic order here is n_angular // 2 == 7, not order 0.
    np.testing.assert_allclose(round_tripped.values, original.values, rtol=1e-4)


def test_the_harmonic_polar_edge_round_trips():
    """The frequency domain's own angular DFT/IDFT edge is likewise an inverse pair."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(2)
    original = FrequencyHarmonicSignal(values=_random_values(rng), grid=grid)

    round_tripped = original.to_angles().to_harmonics()

    np.testing.assert_allclose(round_tripped.values, original.values, atol=1e-10)


# ======================================================================================
# ``to`` composes the full chain
# ======================================================================================


def test_to_composes_the_full_forward_chain_matching_forward_pft():
    """Walking ``SPACE_POLAR`` to ``FREQUENCY_POLAR`` matches ``forward_pft`` exactly."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    f = np.exp(-(grid.r.T**2))
    signal = SpacePolarSignal(values=f, grid=grid)

    walked = signal.to(Domain.FREQUENCY_POLAR)

    expected = forward_pft(f=f, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.domain is Domain.FREQUENCY_POLAR


def test_to_composes_the_full_backward_chain_matching_inverse_pft():
    """Walking ``FREQUENCY_POLAR`` to ``SPACE_POLAR`` matches ``inverse_pft`` exactly."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    F = np.pi * np.exp(-(grid.rho.T**2) / 4.0)
    signal = FrequencyPolarSignal(values=F, grid=grid)

    walked = signal.to(Domain.SPACE_POLAR)

    expected = inverse_pft(F=F, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.domain is Domain.SPACE_POLAR


def test_to_composes_the_full_forward_chain_for_a_3d_batch():
    """``to`` also composes correctly with a trailing batch axis present."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    f = np.broadcast_to(
        array=np.exp(-(grid.r.T**2))[..., np.newaxis], shape=(_N_RADIAL, _N_ANGULAR, 3)
    ).copy()
    signal = SpacePolarSignal(values=f, grid=grid)

    walked = signal.to(Domain.FREQUENCY_POLAR)

    expected = forward_pft(f=f, grid=grid)
    np.testing.assert_allclose(walked.values, expected)
    assert walked.batch_axis == signal.batch_axis


def test_to_a_signals_own_domain_is_a_no_op():
    """Walking to the domain a signal is already in returns it unchanged."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    rng = np.random.default_rng(3)
    signal = SpaceHarmonicSignal(values=_random_values(rng), grid=grid)

    walked = signal.to(Domain.SPACE_HARMONIC)

    np.testing.assert_array_equal(walked.values, signal.values)


def test_to_rejects_a_non_domain_type():
    """The dynamic ``to`` type-validates its ``domain`` argument."""
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal = SpacePolarSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(TypeError):
        signal.to("FREQUENCY_POLAR")  # type: ignore[arg-type]


def test_to_rejects_an_enum_member_of_the_wrong_type():
    """The dynamic ``to`` rejects an enum member that isn't a ``Domain``."""
    from pypft.transform import Direction

    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal = SpacePolarSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(ValueError):
        signal.to(Direction.FORWARD)  # type: ignore[arg-type]


# ======================================================================================
# Illegal edges are pyright errors on hand-written chains
# ======================================================================================


def test_a_hand_written_illegal_edge_is_not_a_valid_attribute():
    """``SpacePolarSignal`` has no ``to_frequency`` -- an illegal, non-adjacent edge.

    The ``# type: ignore[attr-defined]`` below marks exactly what ``pyright`` would
    otherwise reject on this hand-written chain: only the neighbouring subclass
    (``SpaceHarmonicSignal``) defines ``to_frequency``. Removing the comment makes
    the quality gate's own ``pyright`` step fail.
    """
    grid = PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)
    signal: BaseSignal = SpacePolarSignal(
        values=np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex), grid=grid
    )
    with pytest.raises(AttributeError):
        signal.to_frequency()  # type: ignore[attr-defined]
