"""Fixtures for the angular DFT test suite."""

import pytest

from pypft.dft import DFTImplementation

ALL_IMPLEMENTATIONS = tuple(DFTImplementation)

#: Angular sample counts exercised across the parametrized test suite: one
#: odd, one even -- both parities are equally valid.
N_ANGULAR_ODD = 15
N_ANGULAR_EVEN = 16
N_ANGULAR_CASES = (N_ANGULAR_ODD, N_ANGULAR_EVEN)


@pytest.fixture(params=ALL_IMPLEMENTATIONS, ids=lambda impl: impl.name)
def implementation(request: pytest.FixtureRequest) -> DFTImplementation:
    """Each ``DFTImplementation`` strategy, in turn."""
    return request.param


@pytest.fixture(params=N_ANGULAR_CASES, ids=lambda n: f"n_angular={n}")
def n_angular(request: pytest.FixtureRequest) -> int:
    """An odd and an even angular sample count, in turn."""
    return request.param
