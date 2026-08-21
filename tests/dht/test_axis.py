"""Tests for applying the DHT kernel along an arbitrary axis of an N-D array.

``numpy.matmul`` treats only the *last two* dimensions as the matrix and
broadcasts over every leading dimension, so moving the target
axis to ``-2`` (not ``0``) before applying the kernel is what makes
``BaseDHT._apply_along_axis`` correct for every array rank and axis placement.
These tests exercise that generalization directly; ``tests/dht/test_gaussian.py``
and ``tests/dht/test_roundtrip_lena.py`` continue to exercise the pre-existing
1-D, default-axis behavior unchanged.
"""

import numpy as np
import pytest

from pypft.dht import _IMPLEMENTATIONS, hankel_transform
from pypft.dht._naive import NaiveDHT

from .conftest import SIGNAL_SIZE

_ORDER = 0
_R = 1.0

#: ``(shape, axis)`` cases spanning 1-D through 4-D and both axis conventions,
#: with ``SIGNAL_SIZE`` placed at every position in turn.
_SHAPE_AXIS_CASES = [
    ((SIGNAL_SIZE,), 0),
    ((SIGNAL_SIZE,), -1),
    ((SIGNAL_SIZE, 5), 0),
    ((5, SIGNAL_SIZE), -1),
    ((SIGNAL_SIZE, 3, 4), 0),
    ((3, SIGNAL_SIZE, 4), 1),
    ((3, 4, SIGNAL_SIZE), -1),
    ((SIGNAL_SIZE, 2, 3, 4), 0),
    ((2, 3, 4, SIGNAL_SIZE), 3),
]


@pytest.mark.parametrize("shape,axis", _SHAPE_AXIS_CASES, ids=lambda v: str(v))
def test_apply_along_axis_matches_reference(implementation, shape, axis):
    """``_apply_along_axis`` matches ``numpy.apply_along_axis`` at every rank/axis."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(_ORDER, SIGNAL_SIZE)
    rng = np.random.default_rng(0)
    values = rng.standard_normal(shape)

    actual = impl._apply_along_axis(kernel, values, axis)
    expected = np.apply_along_axis(lambda v: kernel @ v, axis, values)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_apply_along_axis_is_bit_identical_on_1d_and_2d(implementation):
    """No behavior change for the 1-D/2-D shapes the suite already exercised."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(_ORDER, SIGNAL_SIZE)
    rng = np.random.default_rng(1)

    vector = rng.standard_normal(SIGNAL_SIZE)
    assert np.array_equal(
        impl._apply_along_axis(kernel, vector, axis=-1),
        impl._apply(kernel, vector),
    )

    batch = rng.standard_normal((SIGNAL_SIZE, 6))
    assert np.array_equal(
        impl._apply_along_axis(kernel, batch, axis=0),
        impl._apply(kernel, batch),
    )


def test_positive_and_negative_axis_indices_agree(implementation):
    """``axis=1`` and the equivalent ``axis=-2`` produce the same result."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(_ORDER, SIGNAL_SIZE)
    rng = np.random.default_rng(2)
    values = rng.standard_normal((3, SIGNAL_SIZE, 4))

    positive = impl._apply_along_axis(kernel, values, axis=1)
    negative = impl._apply_along_axis(kernel, values, axis=-2)

    np.testing.assert_allclose(positive, negative, rtol=1e-10, atol=1e-10)


def test_apply_along_axis_preserves_complex_dtype(implementation):
    """A complex input stays complex through an N-D, non-default-axis application."""
    impl = _IMPLEMENTATIONS[implementation]
    kernel, _ = NaiveDHT._bessel_kernel(_ORDER, SIGNAL_SIZE)
    rng = np.random.default_rng(3)
    values = rng.standard_normal((3, SIGNAL_SIZE, 4)) + 1j * rng.standard_normal(
        (3, SIGNAL_SIZE, 4)
    )

    result = impl._apply_along_axis(kernel, values, axis=1)

    assert result.dtype == np.complex128


def test_hankel_transform_axis_matches_looping_over_other_axes(implementation):
    """``hankel_transform(..., axis=)`` on a batch equals looping the 1-D transform."""
    rng = np.random.default_rng(4)
    batch = rng.standard_normal((SIGNAL_SIZE, 5))

    batched = hankel_transform(batch, _ORDER, _R, implementation, axis=0)

    for i in range(batch.shape[1]):
        expected = hankel_transform(batch[:, i], _ORDER, _R, implementation)
        np.testing.assert_allclose(batched[:, i], expected, rtol=1e-10, atol=1e-10)
