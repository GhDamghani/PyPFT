"""The full polar Fourier transform (PFT) / inverse PFT (IPFT) pipeline.

``forward_pft``/``inverse_pft`` chain the angular DFT/IDFT (``pypft.dft``)
with a per-harmonic, ``R``-scaled discrete Hankel transform (``pypft.dht``),
exactly as ``README.md``'s own diagram: forward goes ``f(r, theta)
--DFT_theta--> f_n(r) --H_n--> F_n(rho) --IDFT_phi--> F(rho, phi)``, and
inverse retraces the same three steps in the opposite direction. This ports
Yao & Baddour's own verified MATLAB appendix (PeerJ CS Part II, Appendix
A-5/A-6) onto ``pypft.dft``'s and ``pypft.dht``'s existing, separately
verified public entry points -- no new numerical kernel is introduced here.

Two facts from the appendix carry over unchanged (see ``scaled_hankel``'s own
docstring for the derivation):

- A negative harmonic order ``n`` never needs its own Hankel-transform
  kernel: ``Y^{(-n)N} = (-1)^n Y^{nN}`` exactly, so ``scaled_hankel`` always
  calls the DHT with ``abs(n)`` and folds the sign in as a scalar.
- The forward step's extra scale factor is ``2*pi * i**(-n)`` on top of
  ``hankel_transform``'s own ``R**2/j_nN``; the inverse step's is
  ``i**n / (2*pi)`` on top of ``inverse_hankel_transform``'s own
  ``j_nN/R**2``.

``forward_pft``/``inverse_pft`` follow PyPFT's own ``(radial, angular)``
array layout (``pypft.axes.Axis``), matching
``pypft.geometry.cartesian_to_polar``'s convention -- **not**
``pypft.grid.PolarGrid.r``'s/``pypft.grid.sample_cartesian``'s own
``(angular, radial)`` layout (each row of ``PolarGrid.r`` is one harmonic's
own radial samples, the natural shape for *building* the grid, not for
storing a transformed image). Transpose a ``sample_cartesian`` result before
passing it to ``forward_pft``.
"""

from enum import Enum, auto

import numpy as np

from pypft.axes import Axis
from pypft.dft import angular_dft, inverse_angular_dft
from pypft.dht import hankel_transform, inverse_hankel_transform
from pypft.grid import PolarGrid, _type_is_polar_grid
from pypft.utils.validators import EnumValidator, IntValidator, NumpyValidator

# ======================================================================================
# The per-harmonic scaled Hankel transform
# ======================================================================================


class Direction(Enum):
    """Which of the DHT's two, oppositely-scaled forms ``scaled_hankel`` applies."""

    FORWARD = auto()
    INVERSE = auto()


def _harmonic_sign(n: int) -> float:
    """Compute the sign folded into a negative-order harmonic's transform.

    ``Y^{(-n)N} = (-1)^n Y^{nN}`` exactly -- the denominator's squared Bessel
    term is unchanged because ``J_{n-1}(j_nk) = -J_{n+1}(j_nk)`` at a zero of
    ``J_n`` -- so a negative order never needs its own kernel: the DHT is
    always called with ``abs(n)``, and this sign is multiplied in afterwards.

    :param n: The (possibly negative) harmonic order.
    :type n: int
    :returns: ``(-1) ** abs(n)`` if ``n`` is negative, else ``1.0``.
    :rtype: float

    """
    return (-1.0) ** abs(n) if n < 0 else 1.0


def scaled_hankel(
    values: np.ndarray,
    grid: PolarGrid,
    *,
    direction: Direction,
    axis: int,
) -> np.ndarray:
    """Apply ``grid``'s per-harmonic, sign- and scale-corrected Hankel transform.

    Loops over ``grid.harmonics`` -- one Hankel transform call per harmonic,
    since each harmonic order needs its own kernel -- along the axis of
    ``values`` complementary to ``axis``, applying ``hankel_transform``/
    ``inverse_hankel_transform`` at that harmonic's ``abs(n)`` and folding in
    ``_harmonic_sign(n)`` plus the direction's own scale factor (forward:
    ``2*pi * i**(-n)``; inverse: ``i**n / (2*pi)`` -- PeerJ CS Part II,
    Appendix A-5/A-6).

    This is the single place in ``src/`` that names the radial axis:
    ``forward_pft``/``inverse_pft`` always pass ``Axis.RADIAL`` explicitly
    rather than relying on a default, since ``hankel_transform`` itself
    defaults its own ``axis`` to ``-1`` for an unrelated, purely conventional
    reason (see ``pypft.axes``'s axis-default tiering).

    :param values: A 2-D array with one axis of length ``grid.n_radial`` (the
        radial axis, named by ``axis``) and the other of length
        ``grid.n_angular``.
    :type values: np.ndarray
    :param grid: The grid whose harmonics, order, and space limit drive the
        per-harmonic transform.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to apply the forward or the inverse scaled
        Hankel transform.
    :type direction: Direction
    :param axis: The axis of ``values`` holding the radial samples.
    :type axis: int
    :returns: ``values`` with the scaled Hankel transform applied along
        ``axis``, one harmonic order at a time.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(value=values)
    NumpyValidator.value_is_2d(value=values)
    _type_is_polar_grid(value=grid)
    EnumValidator.type_is_enum(value=direction)
    EnumValidator.value_is_enum_member(value=direction, enum_class=Direction)
    IntValidator.type_is_int(value=axis)
    NumpyValidator.value_has_axis(value=values, axis=axis)
    # values is 2-D, so the axis complementary to the (validated) radial one
    # is exactly "the other one" -- 1 - radial_axis once axis is normalized.
    radial_axis = axis % values.ndim
    angular_axis = 1 - radial_axis
    NumpyValidator.value1_axis_length_matches_value2(
        value1=values, axis1=angular_axis, value2=grid.harmonics, axis2=0
    )

    result = np.empty(values.shape, dtype=complex)
    for row, harmonic in enumerate(grid.harmonics):
        n = int(harmonic)
        order = abs(n)
        sign = _harmonic_sign(n)
        # Every harmonic's own kernel differs only in `order`, so each row
        # (a single radial line) needs its own hankel_transform call --
        # they cannot be batched together the way a same-order stack can be.
        index: list[slice | int] = [slice(None)] * values.ndim
        index[angular_axis] = row
        index_tuple = tuple(index)
        line = values[index_tuple]
        if direction is Direction.FORWARD:
            factor = sign * (2.0 * np.pi) * (1j ** (-n))
            result[index_tuple] = factor * hankel_transform(f=line, n=order, R=grid.R)
        else:
            factor = sign * (1j**n) / (2.0 * np.pi)
            result[index_tuple] = factor * inverse_hankel_transform(
                F=line, n=order, R=grid.R
            )
    return result


# ======================================================================================
# The forward and inverse PFT
# ======================================================================================


def _validate_pft_input(values: np.ndarray, grid: PolarGrid) -> None:
    """Validate the shared arguments of ``forward_pft``/``inverse_pft``.

    :param values: The polar array to be transformed.
    :type values: np.ndarray
    :param grid: The sampling grid ``values`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is not 2-D or its shape does not match
        ``grid``'s ``(n_radial, n_angular)`` layout.

    """
    NumpyValidator.type_is_ndarray(value=values)
    NumpyValidator.value_is_2d(value=values)
    _type_is_polar_grid(value=grid)
    reference = np.empty((grid.n_radial, grid.n_angular))
    NumpyValidator.value1_shape_matches_value2(value1=values, value2=reference)


def forward_pft(f: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Compute the forward polar Fourier transform (PFT).

    ``f(r, theta) --DFT_theta--> f_n(r) --H_n--> F_n(rho) --IDFT_phi-->
    F(rho, phi)``: an angular DFT turns the physical angle axis into a
    harmonic-order axis, ``scaled_hankel`` applies the order-specific radial
    transform, and an angular IDFT turns the harmonic axis back into a
    physical angle -- this time the frequency domain's own ``phi``.

    :param f: The space-domain samples ``f(r, theta)``, on ``grid``'s
        ``(n_radial, n_angular)`` layout (``pypft.axes.Axis``).
    :type f: np.ndarray
    :param grid: The sampling grid ``f`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :returns: The frequency-domain samples ``F(rho, phi)``, on the same
        layout.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``f`` is not 2-D or its shape does not match
        ``grid``.

    """
    _validate_pft_input(values=f, grid=grid)
    f_n = angular_dft(x=f, axis=Axis.ANGULAR)
    F_n = scaled_hankel(
        values=f_n, grid=grid, direction=Direction.FORWARD, axis=Axis.RADIAL
    )
    return inverse_angular_dft(X=F_n, axis=Axis.ANGULAR)


def inverse_pft(F: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Compute the inverse polar Fourier transform (IPFT).

    The exact mirror of ``forward_pft``: ``F(rho, phi) --DFT_phi--> F_n(rho)
    --H_n--> f_n(r) --IDFT_theta--> f(r, theta)``.

    :param F: The frequency-domain samples ``F(rho, phi)``, on ``grid``'s
        ``(n_radial, n_angular)`` layout (``pypft.axes.Axis``).
    :type F: np.ndarray
    :param grid: The sampling grid ``F`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :returns: The space-domain samples ``f(r, theta)``, on the same layout.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``F`` is not 2-D or its shape does not match
        ``grid``.

    """
    _validate_pft_input(values=F, grid=grid)
    F_n = angular_dft(x=F, axis=Axis.ANGULAR)
    f_n = scaled_hankel(
        values=F_n, grid=grid, direction=Direction.INVERSE, axis=Axis.RADIAL
    )
    return inverse_angular_dft(X=f_n, axis=Axis.ANGULAR)
