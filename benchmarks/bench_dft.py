"""pytest-benchmark suite comparing angular DFT implementations.

Unlike the DHT (two independent optimization axes: kernel build vs. kernel
application), the angular DFT has a single axis -- which FFT library computes
the transform -- so there is one benchmark group per realistic usage pattern
instead of a kernel-build/apply split.

Not part of the package's test suite (outside ``testpaths``); run explicitly
via ``run_dft_benchmarks.py``, which invokes this file directly with
``pytest``.
"""

import numpy as np
import pytest

from pypft.dft import _IMPLEMENTATIONS, DFTImplementation

FORWARD_SIZE = 128

#: Shape/axis for the batched case (the eventual (radial, angular, batch)
#: layout): the angular axis sits in the middle of a 3-D array, exercising
#: ``numpy.fft``/``scipy.fft``'s own axis handling rather than a bare 1-D
#: vector.
BATCH_SHAPE = (32, FORWARD_SIZE, 64)
BATCH_AXIS = 1


@pytest.mark.parametrize("implementation", list(DFTImplementation))
def test_bench_forward_repeated(benchmark, implementation):
    """End-to-end forward angular DFT, called repeatedly at a fixed size.

    Representative of real usage: many radial lines, each a length-``size``
    angular signal, transformed one at a time.
    """
    impl = _IMPLEMENTATIONS[implementation]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(FORWARD_SIZE)
    benchmark(impl.forward, x)


@pytest.mark.parametrize("implementation", list(DFTImplementation))
def test_bench_forward_batched(benchmark, implementation):
    """Forward angular DFT along a non-trailing axis of a batched array.

    The regime that would justify adding ``SCIPY_WORKERS`` (parallel
    ``scipy.fft`` across the batch), if it turns out to win here.
    """
    impl = _IMPLEMENTATIONS[implementation]
    rng = np.random.default_rng(1)
    x = rng.standard_normal(BATCH_SHAPE)
    benchmark(impl.forward, x, axis=BATCH_AXIS)
