"""Round-trip and cross-implementation agreement for the angular DFT."""

import numpy as np

from pypft.dft import DFTImplementation, angular_dft, inverse_angular_dft

from .conftest import N_ANGULAR_CASES


def test_inverse_undoes_forward_at_both_parities(n_angular, implementation):
    """``inverse_angular_dft(angular_dft(x))`` recovers ``x`` at both parities."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n_angular) + 1j * rng.standard_normal(n_angular)

    roundtripped = inverse_angular_dft(
        X=angular_dft(x=x, implementation=implementation), implementation=implementation
    )

    np.testing.assert_allclose(roundtripped, x, rtol=1e-10, atol=1e-10)


def test_implementations_agree_on_a_random_signal(n_angular):
    """Every ``DFTImplementation`` produces the same forward and inverse result."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n_angular) + 1j * rng.standard_normal(n_angular)

    forward_results = [
        angular_dft(x=x, implementation=implementation)
        for implementation in DFTImplementation
    ]
    inverse_results = [
        inverse_angular_dft(X=x, implementation=implementation)
        for implementation in DFTImplementation
    ]

    for result in forward_results[1:]:
        np.testing.assert_allclose(result, forward_results[0], rtol=1e-10, atol=1e-10)
    for result in inverse_results[1:]:
        np.testing.assert_allclose(result, inverse_results[0], rtol=1e-10, atol=1e-10)


def test_angular_dft_axis_matches_looping_over_other_axes(implementation):
    """``angular_dft(..., axis=)`` on a batch equals looping the 1-D transform."""
    n_angular = N_ANGULAR_CASES[0]
    rng = np.random.default_rng(2)
    batch = rng.standard_normal((n_angular, 5)) + 1j * rng.standard_normal(
        (n_angular, 5)
    )

    batched = angular_dft(x=batch, implementation=implementation, axis=0)

    for i in range(batch.shape[1]):
        expected = angular_dft(x=batch[:, i], implementation=implementation)
        np.testing.assert_allclose(batched[:, i], expected, rtol=1e-10, atol=1e-10)
