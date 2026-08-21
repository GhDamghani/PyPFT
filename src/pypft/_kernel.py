"""The explicit, brute-force ``O(N**4)`` polar Fourier transform (PFT) kernel.

``forward_pft``/``inverse_pft`` (``pypft.transform``) compute the PFT as a
composition of three fast, separate steps -- an angular DFT/IDFT
(``pypft.dft``) around a per-harmonic discrete Hankel transform
(``pypft.dht``). Yao & Baddour (PeerJ CS Part II, Eqs. 1-3) instead define
the *same* transform directly, as a single linear operator ``E`` mapping
every flattened input sample to every flattened output sample:

    F_{qm} = sum_{p,k} f_{pk} * E^{-}_{qm;pk}
    f_{pk} = sum_{q,m} F_{qm} * E^{+}_{qm;pk}

with ``E^{-}``/``E^{+}`` themselves a sum over harmonic order ``n`` of a
radial factor (the DHT kernel ``Y^{|n|N}``, scaled exactly as
``pypft.transform.scaled_hankel`` scales it) and an angular phase factor.
Building this ``(n_radial*n_angular, n_radial*n_angular)`` matrix costs
``O(n_angular * n_radial**2)`` to assemble and touches every one of its
``O((n_radial*n_angular)**2)`` entries -- quartic in a single grid
dimension -- which is exactly why ``forward_pft``/``inverse_pft`` never
build it: this module exists purely as a from-scratch, independent oracle
for testing that composition against, not as a faster or more convenient
public entry point (hence the leading underscore -- unlike ``pypft.dft``,
this is not meant for everyday use, only for verification).

Deliberately not a ``DHTImplementation``-style enum of interchangeable
strategies: there is exactly one way to build this matrix, so a Strategy
pattern would add indirection without a second implementation to justify
it.
"""

import numpy as np

from pypft.dft import harmonics
from pypft.dht._naive import NaiveDHT
from pypft.grid import PolarGrid, _type_is_polar_grid
from pypft.transform import Direction
from pypft.utils.validators import EnumValidator


def _harmonic_sign(n: int) -> float:
    """Compute the sign relating a negative harmonic's kernel to its positive twin.

    Duplicated from ``pypft.transform`` rather than imported: this module's
    whole purpose is to be an *independent* check of that module's
    composition, so re-using its internals here would silently hide a bug
    common to both.

    :param n: The (possibly negative) harmonic order.
    :type n: int
    :returns: ``(-1) ** abs(n)`` if ``n`` is negative, else ``1.0``.
    :rtype: float

    """
    return (-1.0) ** abs(n) if n < 0 else 1.0


def kernel_matrix(grid: PolarGrid, *, direction: Direction) -> np.ndarray:
    """Build the explicit ``E^{-}``/``E^{+}`` operator matrix (Eqs. 1-3).

    For every harmonic order ``n`` in ``grid.harmonics``, ``NaiveDHT`` builds
    that order's own ``(n_radial, n_radial)`` kernel from first principles
    (bypassing ``pypft.dht``'s cache, again to stay independent of the code
    under test), which is combined with an ``(n_angular, n_angular)``
    angular phase factor via a Kronecker product -- the direct matrix form
    of "one Hankel transform per harmonic, wrapped by an angular DFT/IDFT"
    that ``forward_pft``/``inverse_pft`` instead compute step by step.

    :param grid: The sampling grid the transform is defined on.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to build the forward (``E^{-}``) or inverse
        (``E^{+}``) operator.
    :type direction: pypft.transform.Direction
    :returns: The ``(n_radial*n_angular, n_radial*n_angular)`` operator
        matrix. Applying it to a raveled, ``(n_radial, n_angular)``-shaped
        input and reshaping the result back reproduces ``forward_pft``/
        ``inverse_pft`` exactly, up to floating-point noise.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``direction`` is not a ``Direction`` member.

    """
    _type_is_polar_grid(grid)
    EnumValidator.type_is_enum(direction)
    EnumValidator.value_is_enum_member(direction, Direction)

    n_radial = grid.n_radial
    n_angular = grid.n_angular
    theta = grid.theta
    total_size = n_radial * n_angular
    operator = np.zeros((total_size, total_size), dtype=complex)

    for harmonic in harmonics(n_angular):
        n = int(harmonic)
        order = abs(n)
        sign = _harmonic_sign(n)
        # NaiveDHT is the from-scratch reference kernel builder -- every
        # other DHT implementation is verified against it, never the other
        # way around, so it is the right ground truth for an oracle too.
        bessel_kernel, zeros = NaiveDHT._bessel_kernel(order, n_radial)
        j_nN1 = zeros[-1]

        # phase[i] = exp(i*n*theta_i); the angular factor at (row, col) is
        # phase[row] * conj(phase[col]) -- Kronecker-multiplying it against
        # the radial factor below reproduces both E^{-}_{qm;pk}'s (q, p)
        # dependence and E^{+}_{qm;pk}'s (p, q) dependence, since the two
        # differ only in which flattened axis (output or input) plays the
        # "row"/"col" role of this same, direction-agnostic matrix.
        phase = np.exp(1j * n * theta)
        angular = np.outer(phase, phase.conj())

        if direction is Direction.FORWARD:
            # Matches scaled_hankel's forward scale factor exactly: sign *
            # 2*pi * i**(-n) on top of hankel_transform's own R**2/j_nN;
            # the 1/n_angular below is inverse_angular_dft's own
            # normalization (angular_dft itself is unnormalized).
            radial = sign * (2.0 * np.pi) * (1j ** (-n)) * (grid.R**2 / j_nN1)
            operator += np.kron(radial * bessel_kernel, angular) / n_angular
        else:
            radial = sign * (1j**n) * (j_nN1 / grid.R**2) * bessel_kernel
            operator += np.kron(radial, angular) / (2.0 * np.pi * n_angular)

    return operator
