"""pytest-benchmark suite comparing ``PFTImplementation`` strategies.

``scaled_hankel``'s per-harmonic loop can be carried out two ways --
``HARMONIC_LOOP`` (one ``hankel_transform`` call per harmonic, Python-level)
or ``STACKED_KERNEL`` (every harmonic's kernel stacked into one array,
applied with a single batched ``matmul``). The batched scenario below (a
3-D ``(radial, angular, batch)`` array, PyPFT's own batching layout) is the
one that actually drives ``DEFAULT_PFT_IMPLEMENTATION``, since that is
exactly the regime a Python-level per-harmonic loop pays repeated call
overhead for and a single batched ``matmul`` does not.

Not part of the package's test suite (outside ``testpaths``); run explicitly
via ``run_pft_benchmarks.py``, which invokes this file directly with
``pytest``.
"""

import numpy as np
import pytest

from pypft.grid import PolarGrid
from pypft.transform import Direction, PFTImplementation, scaled_hankel

#: A grid with enough harmonics to make the Python-level per-harmonic loop's
#: own call overhead visible against a single batched ``matmul``.
_GRID = PolarGrid(n_radial=128, n_angular=31, R=1.0)

BATCH_SIZE = 64


def _random_values(rng: np.random.Generator, batch: int | None) -> np.ndarray:
    shape: tuple[int, ...] = (_GRID.n_radial, _GRID.n_angular)
    if batch is not None:
        shape += (batch,)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("implementation", list(PFTImplementation))
def test_bench_scaled_hankel_repeated(benchmark, implementation):
    """The plain 2-D case, called repeatedly at a fixed grid size."""
    rng = np.random.default_rng(0)
    values = _random_values(rng, batch=None)
    benchmark(
        scaled_hankel,
        values=values,
        grid=_GRID,
        direction=Direction.FORWARD,
        axis=0,
        angular_axis=1,
        implementation=implementation,
    )


@pytest.mark.parametrize("implementation", list(PFTImplementation))
def test_bench_scaled_hankel_batched(benchmark, implementation):
    """The 3-D ``(radial, angular, batch)`` case PyPFT's own batching targets."""
    rng = np.random.default_rng(1)
    values = _random_values(rng, batch=BATCH_SIZE)
    benchmark(
        scaled_hankel,
        values=values,
        grid=_GRID,
        direction=Direction.FORWARD,
        axis=0,
        angular_axis=1,
        implementation=implementation,
    )
