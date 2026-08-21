"""pytest-benchmark suite comparing DHT implementations along both optimization
axes of the transform's discretization scheme: computing the Bessel-valued
kernel, and applying that kernel to a signal.

Not part of the package's test suite (outside ``testpaths``); run explicitly via
``run_dht_benchmarks.py``, which invokes this file directly with ``pytest``.
"""

import numpy as np
import pytest

from pypft.dht import _IMPLEMENTATIONS, DHTImplementation
from pypft.dht._naive import NaiveDHT

KERNEL_ORDER = 0
KERNEL_SIZE = 128

FORWARD_ORDER = 0
FORWARD_SIZE = 256
FORWARD_R = 1.0

APPLY_BATCH = 64

#: Shape/axis for the N-D/batched case: a (radial, angular, batch)-shaped
#: array with the transform axis first, the layout ``pypft.transform`` will
#: actually call the DHT with.
ND_SHAPE = (FORWARD_SIZE, 15, APPLY_BATCH)
ND_AXIS = 0


def test_bench_kernel_build(benchmark):
    """Axis (a): building the (uncached) kernel from scratch."""
    benchmark(NaiveDHT._bessel_kernel, KERNEL_ORDER, KERNEL_SIZE)


@pytest.mark.parametrize("implementation", list(DHTImplementation))
def test_bench_forward_repeated(benchmark, implementation):
    """End-to-end forward transform, called repeatedly at a fixed (n, size).

    Representative of real usage (e.g. many radial lines sharing one order and
    size): implementations with a kernel cache pay the kernel-build cost once.
    """
    impl = _IMPLEMENTATIONS[implementation]
    rng = np.random.default_rng(0)
    f = rng.standard_normal(FORWARD_SIZE)
    benchmark(impl.forward, f, FORWARD_ORDER, FORWARD_R)


@pytest.mark.parametrize(
    "implementation", [DHTImplementation.NAIVE, DHTImplementation.VECTORIZED]
)
def test_bench_apply_batch(benchmark, implementation):
    """Axis (b): applying a pre-built kernel to a batch of signals at once."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(FORWARD_ORDER, FORWARD_SIZE)
    rng = np.random.default_rng(1)
    batch = rng.standard_normal((FORWARD_SIZE, APPLY_BATCH))
    impl._apply(kernel, batch)  # warm up (numba JIT compilation, if applicable)
    benchmark(impl._apply, kernel, batch)


@pytest.mark.parametrize("implementation", list(DHTImplementation))
def test_bench_apply_along_axis_nd(benchmark, implementation):
    """The N-D/batched case: axis 0 of a (radial, angular, batch) array.

    This is the first workload exercising ``VECTORIZED``'s mat-mat regime via
    ``_apply_batched`` (a flatten-to-2D reshape around ``_matmat_parallel``)
    instead of a plain vector, so it can change which implementation wins.
    """
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(FORWARD_ORDER, FORWARD_SIZE)
    rng = np.random.default_rng(2)
    values = rng.standard_normal(ND_SHAPE)
    impl._apply_along_axis(kernel, values, ND_AXIS)  # warm up numba, if applicable
    benchmark(impl._apply_along_axis, kernel, values, ND_AXIS)
