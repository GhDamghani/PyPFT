"""Tests for the PFT's own analytical properties (Baddour Part I, Mathematics 7(8):698).

These come from the same paper that defines the 2-D DFT-in-polar-coordinates
kernel ``E^{-}``/``E^{+}`` this package's ``pypft._kernel.kernel_matrix``
builds directly (Eqs. 1-3 of the companion PeerJ CS Part II paper -- Part I
restates the same kernel as its own Eq. 32). Properties that are purely
about the kernel (orthogonality, the delta pair, the generalized shift, and
the rules built from it) are tested against that explicit operator, since it
is independent of ``forward_pft``/``inverse_pft``'s own FFT-based
composition; properties about the *transform itself* (rotation equivariance,
linearity, the DC term, conjugate symmetry) are tested directly against
``forward_pft``/``inverse_pft``, which is both simpler and closer to how a
caller would observe them.

The paper additionally shows a Parseval relationship for an alternative,
*symmetric* kernel choice (Eqs. 78-79) that this package does not implement
anywhere -- the same "non-symmetric ``Y^{nN}`` vs. symmetric ``T^{nN}``"
choice already made at the DHT level (``src/pypft/dht/_base.py``) carries
through to the PFT level. Building a second, otherwise-unused kernel just to
exercise that one relationship would add code with no other purpose, so only
the non-symmetric-kernel Parseval relationship (Eqs. 80, 87-88), which *is*
the kernel this package actually uses, is tested below.
"""

import numpy as np

from pypft._kernel import kernel_matrix
from pypft.axes import Axis
from pypft.dft import angular_dft, harmonics
from pypft.dht import hankel_transform
from pypft.grid import PolarGrid
from pypft.transform import Direction, forward_pft, inverse_pft, scaled_hankel

#: Small enough to keep the O(n_radial**2 * n_angular**3) explicit kernel
#: fast to build, while spanning several distinct harmonic orders (0-2).
_GRID = PolarGrid(n_radial=6, n_angular=5, R=3.0)

#: The explicit kernel's own residual (from ``Y@Y != I`` exactly in floating
#: point, summed over several harmonics) is larger than the DHT's own
#: per-order tolerance, since these tests combine multiple orders at once.
_KERNEL_TOL = 1e-3


def _flat_shift(
    apply_operator: np.ndarray,
    column_operator: np.ndarray,
    transform: np.ndarray,
    index0: int,
) -> np.ndarray:
    """Compute the PFT's generalized shift of a flattened vector (Eqs. 46-49).

    Mirrors the DHT's own generalized shift
    (``tests/dht/test_kernel_properties.py``'s ``_generalized_shift``):
    ``shift(V, idx0) := apply_operator @ (column_operator[:, idx0] * V)``.
    Unlike the DHT's self-inverse ``Y`` (one kernel serves both roles), the
    PFT's forward and inverse operators are *different* matrices, so which
    one modulates by its ``idx0`` column and which one is applied afterwards
    depends on which domain ``transform`` already lives in: shifting a
    frequency-domain vector into a space-domain result uses the inverse
    operator to apply and the forward operator's column (and vice versa).

    :param apply_operator: The ``kernel_matrix`` applied to the modulated
        vector, mapping into the shifted result's own domain.
    :type apply_operator: np.ndarray
    :param column_operator: The ``kernel_matrix`` whose ``index0`` column
        modulates ``transform`` before ``apply_operator`` is applied.
    :type column_operator: np.ndarray
    :param transform: The flattened transform of the vector being shifted.
    :type transform: np.ndarray
    :param index0: The flattened shift-amount index.
    :type index0: int
    :returns: The generalized-shifted, flattened vector.
    :rtype: np.ndarray

    """
    return apply_operator @ (column_operator[:, index0] * transform)


def test_pft_kernel_orthogonality_over_frequency_indices():
    """``sum_{q,m} E^-_{qm;pk} E^+_{qm;p'k'} = delta_pp' delta_kk'`` (Eq. 34).

    In matrix form this is exactly ``K_inverse @ K_forward == I``.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    identity = np.eye(forward.shape[0])
    np.testing.assert_allclose(
        inverse @ forward, identity, rtol=_KERNEL_TOL, atol=_KERNEL_TOL
    )


def test_pft_kernel_orthogonality_over_spatial_indices():
    """``sum_{p,k} E^-_{qm;pk} E^+_{q'm';pk} = delta_qq' delta_mm'`` (Eq. 37).

    The mirror image of the frequency-index orthogonality above:
    ``K_forward @ K_inverse == I``.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    identity = np.eye(forward.shape[0])
    np.testing.assert_allclose(
        forward @ inverse, identity, rtol=_KERNEL_TOL, atol=_KERNEL_TOL
    )


def test_pft_complex_exponential_transforms_to_a_delta():
    """``f_{pk} = E^+_{q0,m0;pk}`` transforms to a delta at ``(q0, m0)`` (Eq. 43).

    ``E^+_{q0,m0;pk}``, viewed as a function of ``(p, k)`` at a fixed
    ``(q0, m0)``, is exactly one column of the inverse kernel matrix.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    index0 = 8

    f = inverse[:, index0]
    F = forward @ f
    delta = np.zeros_like(F)
    delta[index0] = 1.0
    np.testing.assert_allclose(F, delta, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)


def test_pft_shift_modulation_rule():
    """The forward transform of a generalized-shifted spectrum is a modulation (Eqs. 46-53).

    ``F^{2D}(f^{p0,k0}) == F_{qm} * E^-_{qm;p0k0}``, where ``f^{p0,k0}`` is
    the shift of the vector whose forward transform is ``F``.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    rng = np.random.default_rng(0)
    F = rng.standard_normal(forward.shape[0]) + 1j * rng.standard_normal(
        forward.shape[0]
    )
    index0 = 7

    shifted = _flat_shift(
        apply_operator=inverse,
        column_operator=forward,
        transform=F,
        index0=index0,
    )
    lhs = forward @ shifted
    rhs = forward[:, index0] * F
    np.testing.assert_allclose(lhs, rhs, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)


def test_pft_modulation_rule():
    """Modulating in space by an ``E^+`` column is a generalized shift in frequency (Eq. 55-62).

    ``f_{pk} = E^+_{q0,m0;pk} * g_{pk}`` transforms to the generalized
    shift, in frequency, of ``g``'s own spectrum.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    rng = np.random.default_rng(1)
    g = rng.standard_normal(forward.shape[0]) + 1j * rng.standard_normal(
        forward.shape[0]
    )
    index0 = 7

    modulated = inverse[:, index0] * g
    lhs = forward @ modulated
    rhs = _flat_shift(
        apply_operator=forward,
        column_operator=inverse,
        transform=g,
        index0=index0,
    )
    np.testing.assert_allclose(lhs, rhs, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)


def test_pft_convolution_multiplication_rule():
    """The forward transform of a generalized convolution is a Hadamard product (Sec. 6.6, Eq. 68).

    ``f = h ** g := sum_idx0 h^{shift}_{idx0} * g_{idx0}`` transforms to
    ``H * G`` entrywise; also checks the convolution is commutative.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    size = forward.shape[0]
    rng = np.random.default_rng(2)
    g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    h = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    G = forward @ g
    H = forward @ h

    zero = np.zeros(size, dtype=complex)
    convolution = sum(
        (
            _flat_shift(
                apply_operator=inverse,
                column_operator=forward,
                transform=H,
                index0=idx0,
            )
            * g[idx0]
            for idx0 in range(size)
        ),
        start=zero,
    )
    swapped = sum(
        (
            _flat_shift(
                apply_operator=inverse,
                column_operator=forward,
                transform=G,
                index0=idx0,
            )
            * h[idx0]
            for idx0 in range(size)
        ),
        start=zero,
    )
    np.testing.assert_allclose(convolution, swapped, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)

    lhs = forward @ convolution
    rhs = H * G
    np.testing.assert_allclose(lhs, rhs, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)


def test_pft_multiplication_convolution_rule():
    """Multiplication in space is a generalized convolution in frequency (Sec. 6.7, Eq. 71).

    ``f = h * g`` (entrywise, space domain) transforms to
    ``sum_idx0 H[idx0] * shift(g, idx0)`` -- the mirror of the convolution
    rule above; also checks this frequency-domain convolution commutes.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    size = forward.shape[0]
    rng = np.random.default_rng(3)
    g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    h = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    G = forward @ g
    H = forward @ h

    f = h * g
    F = forward @ f
    zero = np.zeros(size, dtype=complex)
    freq_convolution = sum(
        (
            H[idx0]
            * _flat_shift(
                apply_operator=forward,
                column_operator=inverse,
                transform=g,
                index0=idx0,
            )
            for idx0 in range(size)
        ),
        start=zero,
    )
    np.testing.assert_allclose(F, freq_convolution, rtol=_KERNEL_TOL, atol=_KERNEL_TOL)

    swapped = sum(
        (
            G[idx0]
            * _flat_shift(
                apply_operator=forward,
                column_operator=inverse,
                transform=h,
                index0=idx0,
            )
            for idx0 in range(size)
        ),
        start=zero,
    )
    np.testing.assert_allclose(
        freq_convolution, swapped, rtol=_KERNEL_TOL, atol=_KERNEL_TOL
    )


def test_pft_generalized_parseval_non_symmetric_kernel():
    """The starred-conjugate Parseval relationship holds for this package's own kernel (Eqs. 80, 87-88).

    The plain conjugate does not preserve the inner product across domains
    for the non-symmetric ``E^-``/``E^+`` kernel this package uses (only the
    *symmetric* kernel choice this package does not implement gets that, per
    this module's own docstring) -- instead, a "starred" conjugate built
    from the *opposite*-direction kernel is required: ``conj*(g) :=
    K_forward.T @ conj(forward(g))`` for a space-domain ``g``, and
    ``conj*(G) := K_inverse.T @ conj(inverse(G))`` for a frequency-domain
    ``G``.
    """
    forward = kernel_matrix(grid=_GRID, direction=Direction.FORWARD)
    inverse = kernel_matrix(grid=_GRID, direction=Direction.INVERSE)
    size = forward.shape[0]
    rng = np.random.default_rng(4)
    g = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    h = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    G = forward @ g
    H = forward @ h

    conj_star_freq = inverse.T @ np.conj(
        g
    )  # conj*(g), as a function of frequency index
    conj_star_space = forward.T @ np.conj(G)  # conj*(G), as a function of space index

    lhs_freq_domain = np.sum(H * conj_star_freq)
    rhs_freq_domain = np.sum(h * np.conj(g))
    np.testing.assert_allclose(
        lhs_freq_domain, rhs_freq_domain, rtol=_KERNEL_TOL, atol=_KERNEL_TOL
    )

    lhs_space_domain = np.sum(h * conj_star_space)
    rhs_space_domain = np.sum(H * np.conj(G))
    np.testing.assert_allclose(
        lhs_space_domain, rhs_space_domain, rtol=_KERNEL_TOL, atol=_KERNEL_TOL
    )


def test_forward_pft_is_linear():
    """``forward_pft`` is a linear map: scaling and superposition commute with it."""
    rng = np.random.default_rng(5)
    f1 = rng.standard_normal((_GRID.n_radial, _GRID.n_angular))
    f2 = rng.standard_normal((_GRID.n_radial, _GRID.n_angular))
    a, b = 2.3, -1.7

    lhs = forward_pft(f=a * f1 + b * f2, grid=_GRID)
    rhs = a * forward_pft(f=f1, grid=_GRID) + b * forward_pft(f=f2, grid=_GRID)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


def test_forward_pft_of_a_circularly_symmetric_signal_is_a_bare_hankel_transform():
    """A circularly symmetric input's PFT is the plain order-0 Hankel transform.

    Only the harmonic-0 term of the angular decomposition survives a
    function that does not depend on angle, so the entire DFT/DHT/IDFT
    chain collapses to a single order-0 Hankel transform, scaled by
    ``2*pi`` -- the harmonic-0 term's own scale factor
    (``pypft.transform.scaled_hankel``) -- constant across every angular
    column of the output. This is the PFT's analogue of the classical
    continuous 2-D Fourier transform's "mean"/DC term.
    """
    rng = np.random.default_rng(6)
    radial_profile = rng.standard_normal(_GRID.n_radial)
    f = np.broadcast_to(
        array=radial_profile[:, np.newaxis], shape=(_GRID.n_radial, _GRID.n_angular)
    )

    F = forward_pft(f=f, grid=_GRID)
    expected_column = 2.0 * np.pi * hankel_transform(f=radial_profile, n=0, R=_GRID.R)
    np.testing.assert_allclose(
        F,
        np.broadcast_to(array=expected_column[:, np.newaxis], shape=F.shape),
        rtol=1e-10,
        atol=1e-10,
    )


def test_forward_pft_rotation_equivariance():
    """Rotating the frequency domain by ``q0`` is equivalent to rotating the space domain (Eq. 75).

    ``inverse_pft`` of a circularly-shifted spectrum equals the
    circularly-shifted space-domain reconstruction, at every shift amount
    including the harmonic-range edges -- a global angular roll is
    invariant to which physical angle is called "index 0", so this alone
    cannot pin down the angular origin (that is instead
    ``tests/test_geometry.py``'s wedge test's job).
    """
    rng = np.random.default_rng(7)
    F = rng.standard_normal(
        (_GRID.n_radial, _GRID.n_angular)
    ) + 1j * rng.standard_normal((_GRID.n_radial, _GRID.n_angular))
    f = inverse_pft(F=F, grid=_GRID)

    for shift in range(_GRID.n_angular):
        lhs = inverse_pft(F=np.roll(a=F, shift=shift, axis=Axis.ANGULAR), grid=_GRID)
        rhs = np.roll(a=f, shift=shift, axis=Axis.ANGULAR)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)


def test_forward_pft_angular_spectrum_has_twisted_conjugate_symmetry():
    """A real space-domain signal's per-harmonic spectrum is conjugate symmetric, up to a sign.

    ``F_{-n} == (-1)^n * conj(F_n)`` at the harmonic-indexed stage between
    the angular DFT and the angular IDFT -- the ordinary real-DFT conjugate
    symmetry (``f_{-n} == conj(f_n)``), carried through
    ``scaled_hankel``'s real-valued kernel and its own negative-order sign
    relation (``pypft.transform``'s own ``Y^{(-n)N} = (-1)^n * Y^{nN}``).
    This ``(-1)^n`` twist means the *physical* frequency-domain array
    ``F(rho, phi)`` itself is not simply conjugate symmetric in ``phi``, as
    already documented for the Gaussian oracle (``tests/test_transform.py``,
    ``max|imag(F)|`` being nonzero is expected).
    """
    rng = np.random.default_rng(8)
    f = rng.standard_normal((_GRID.n_radial, _GRID.n_angular))
    f_n = angular_dft(x=f, axis=Axis.ANGULAR)
    F_n = scaled_hankel(
        values=f_n,
        grid=_GRID,
        direction=Direction.FORWARD,
        axis=Axis.RADIAL,
        angular_axis=Axis.ANGULAR,
    )

    harmonic_orders = harmonics(n_angular=_GRID.n_angular)
    for index, n in enumerate(harmonic_orders):
        negative_index = list(harmonic_orders).index(-int(n))
        lhs = F_n[:, negative_index]
        rhs = ((-1.0) ** int(n)) * np.conj(F_n[:, index])
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
