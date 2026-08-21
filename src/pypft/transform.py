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

``forward_pft``/``inverse_pft`` follow PyPFT's own ``(radial, angular[,
batch])`` array layout (``pypft.axes.Axis``), matching
``pypft.geometry.cartesian_to_polar``'s convention -- **not**
``pypft.grid.PolarGrid.r``'s/``pypft.grid.sample_cartesian``'s own
``(angular, radial)`` layout (each row of ``PolarGrid.r`` is one harmonic's
own radial samples, the natural shape for *building* the grid, not for
storing a transformed image). Transpose a ``sample_cartesian`` result before
passing it to ``forward_pft``.

Both directions also accept a 3-D ``(radial, angular, batch)`` array, batch
axis last (``pypft.axes.DEFAULT_BATCH_AXIS``) -- the angular DFT/IDFT already
operate along a single named axis of an otherwise arbitrary-rank array
(``pypft.dft``), and ``scaled_hankel`` (below) generalizes the same way, so
batching costs no new numerical kernel either.
"""

from enum import Enum, auto
from functools import lru_cache
from typing import Callable

import numpy as np

from pypft.axes import DEFAULT_BATCH_AXIS, Axis
from pypft.dft import angular_dft, inverse_angular_dft
from pypft.dht import hankel_transform, inverse_hankel_transform
from pypft.dht._cached import CachedBesselDHT
from pypft.grid import PolarGrid, _type_is_polar_grid
from pypft.utils.validators import EnumValidator, IntValidator, NumpyValidator

# ======================================================================================
# The per-harmonic scaled Hankel transform
# ======================================================================================


class Direction(Enum):
    """Which of the DHT's two, oppositely-scaled forms ``scaled_hankel`` applies."""

    FORWARD = auto()
    INVERSE = auto()


class PFTImplementation(Enum):
    """Selectable strategies for ``scaled_hankel``'s per-harmonic application.

    Both are pure NumPy, differing only in how the harmonic loop itself is
    carried out -- see ``_scaled_hankel_harmonic_loop``/
    ``_scaled_hankel_stacked_kernel`` for the two algorithms.
    """

    HARMONIC_LOOP = auto()
    STACKED_KERNEL = auto()


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


def _scaled_hankel_harmonic_loop(
    values: np.ndarray,
    grid: PolarGrid,
    *,
    direction: Direction,
    axis: int,
    angular_axis: int,
) -> np.ndarray:
    """Apply ``scaled_hankel`` one ``hankel_transform`` call per harmonic.

    The straightforward algorithm: for each harmonic row, slice it out along
    ``angular_axis`` (collapsing that axis by one), then delegate to
    ``hankel_transform``/``inverse_hankel_transform`` -- which are themselves
    already N-D-capable along ``axis``, so a trailing batch axis (if any)
    rides along for free via their own ``matmul`` broadcasting.

    :param values: A 2-D or 3-D array; see ``scaled_hankel``.
    :type values: np.ndarray
    :param grid: The grid whose harmonics, order, and space limit drive the
        per-harmonic transform.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to apply the forward or the inverse scaled
        Hankel transform.
    :type direction: Direction
    :param axis: The (already-validated) axis of ``values`` holding the
        radial samples.
    :type axis: int
    :param angular_axis: The (already-validated) axis of ``values`` holding
        the harmonic samples.
    :type angular_axis: int
    :returns: ``values`` with the scaled Hankel transform applied along
        ``axis``, one harmonic order at a time.
    :rtype: np.ndarray

    """
    radial_axis = axis % values.ndim
    angular_axis = angular_axis % values.ndim
    result = np.empty(values.shape, dtype=complex)
    for row, harmonic in enumerate(grid.harmonics):
        n = int(harmonic)
        order = abs(n)
        sign = _harmonic_sign(n)
        # Every harmonic's own kernel differs only in `order`, so each row
        # (a single radial line, or radial-by-batch plane) needs its own
        # hankel_transform call -- they cannot be batched together the way a
        # same-order stack can be.
        index: list[slice | int] = [slice(None)] * values.ndim
        index[angular_axis] = row
        index_tuple = tuple(index)
        line = values[index_tuple]
        # Collapsing angular_axis (an integer index) removes that axis, so
        # every axis after it -- radial included, if it comes later -- shifts
        # down by one position in `line`.
        line_axis = radial_axis if radial_axis < angular_axis else radial_axis - 1
        if direction is Direction.FORWARD:
            factor = sign * (2.0 * np.pi) * (1j ** (-n))
            result[index_tuple] = factor * hankel_transform(
                f=line, n=order, R=grid.R, axis=line_axis
            )
        else:
            factor = sign * (1j**n) / (2.0 * np.pi)
            result[index_tuple] = factor * inverse_hankel_transform(
                F=line, n=order, R=grid.R, axis=line_axis
            )
    return result


#: Maximum number of distinct ``(grid, direction)`` kernel stacks kept alive at
#: once -- mirrors ``pypft.dht._cached.KERNEL_CACHE_MAXSIZE``'s own reasoning:
#: a stack is ``n_angular`` times one DHT kernel's own size, so an unbounded
#: cache risks unbounded growth across a long-running process cycling through
#: many grids.
STACKED_KERNEL_CACHE_MAXSIZE = 32


@lru_cache(maxsize=STACKED_KERNEL_CACHE_MAXSIZE)
def _stacked_kernel_and_factors(
    grid: PolarGrid, direction: Direction
) -> tuple[np.ndarray, np.ndarray]:
    """Build one ``(n_angular, n_radial, n_radial)`` kernel stack, plus its factors.

    Every harmonic's radial sample count is the same (``grid.n_radial``), so
    every harmonic's kernel shares one ``(n_radial, n_radial)`` shape and can
    be stacked along a new leading axis -- reusing ``CachedBesselDHT``'s own
    ``(n, size)`` LRU cache directly, bypassing ``hankel_transform``'s public
    dispatch and per-call validation, since ``STACKED_KERNEL`` exists
    specifically to remove Python-level per-harmonic overhead. ``PolarGrid``
    is itself frozen and hashable, so this whole stack -- not just the
    individual per-harmonic kernels -- is cached keyed on ``(grid,
    direction)``, the same "many calls share one discretization" usage
    pattern that justifies ``CachedBesselDHT``'s own cache: without this,
    ``STACKED_KERNEL`` measured *slower* than ``HARMONIC_LOOP`` even on a
    batched workload, because rebuilding and copying the whole stack on every
    call dominated the batched ``matmul``'s own saving.

    :param grid: The grid whose harmonics, order, and space limit drive the
        per-harmonic transform.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to apply the forward or the inverse scaled
        Hankel transform.
    :type direction: Direction
    :returns: The stacked kernel (real-valued, one ``(n_radial, n_radial)``
        Bessel kernel per harmonic) and the complex per-harmonic scale
        factor (sign, ``i**(+-n)``, ``2*pi``, and the physical ``R**2/j_nN``
        or ``j_nN/R**2`` term), each indexed the same way as ``grid.harmonics``,
        both marked read-only since they are shared across every caller.
    :rtype: tuple[np.ndarray, np.ndarray]

    """
    size = grid.n_radial
    kernels = np.empty((grid.n_angular, size, size))
    factors = np.empty(grid.n_angular, dtype=complex)
    for row, harmonic in enumerate(grid.harmonics):
        n = int(harmonic)
        order = abs(n)
        sign = _harmonic_sign(n)
        kernel, zeros = CachedBesselDHT._bessel_kernel(n=order, size=size)
        j_nN = zeros[-1]
        kernels[row] = kernel
        if direction is Direction.FORWARD:
            factors[row] = sign * (2.0 * np.pi) * (1j ** (-n)) * (grid.R**2 / j_nN)
        else:
            factors[row] = sign * (1j**n) / (2.0 * np.pi) * (j_nN / grid.R**2)
    kernels.setflags(write=False)
    factors.setflags(write=False)
    return kernels, factors


def _scaled_hankel_stacked_kernel(
    values: np.ndarray,
    grid: PolarGrid,
    *,
    direction: Direction,
    axis: int,
    angular_axis: int,
) -> np.ndarray:
    """Apply ``scaled_hankel`` via a single batched ``matmul`` across every harmonic.

    Stacks every harmonic's kernel into one ``(n_angular, n_radial,
    n_radial)`` array (``_stacked_kernel_and_factors``), moves ``values`` so
    the harmonic axis leads and the radial axis follows (any remaining batch
    axis trails, unmoved -- ``numpy.moveaxis`` with paired ``source``/
    ``destination`` sequences handles this for either a 2-D or a 3-D input in
    one call), and applies the whole stack with one ``numpy.matmul`` call --
    a single batched BLAS ``gemm`` over every harmonic (and batch element)
    at once, instead of ``_scaled_hankel_harmonic_loop``'s one Python-level
    call per harmonic.

    :param values: A 2-D or 3-D array; see ``scaled_hankel``.
    :type values: np.ndarray
    :param grid: The grid whose harmonics, order, and space limit drive the
        per-harmonic transform.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to apply the forward or the inverse scaled
        Hankel transform.
    :type direction: Direction
    :param axis: The (already-validated) axis of ``values`` holding the
        radial samples.
    :type axis: int
    :param angular_axis: The (already-validated) axis of ``values`` holding
        the harmonic samples.
    :type angular_axis: int
    :returns: ``values`` with the scaled Hankel transform applied along
        ``axis``, every harmonic order at once.
    :rtype: np.ndarray

    """
    kernels, factors = _stacked_kernel_and_factors(grid=grid, direction=direction)
    has_batch_axis = values.ndim == 3
    moved = np.moveaxis(a=values, source=(angular_axis, axis), destination=(0, 1))
    if not has_batch_axis:
        moved = moved[..., np.newaxis]  # a length-1 pseudo-batch axis for matmul
    applied = (
        kernels @ moved
    )  # (n_angular, n_radial, n_radial) @ (n_angular, n_radial, batch)
    applied = applied * factors[:, np.newaxis, np.newaxis]
    if not has_batch_axis:
        applied = applied[..., 0]
    return np.moveaxis(a=applied, source=(0, 1), destination=(angular_axis, axis))


_PFT_IMPLEMENTATIONS: dict[PFTImplementation, Callable[..., np.ndarray]] = {
    PFTImplementation.HARMONIC_LOOP: _scaled_hankel_harmonic_loop,
    PFTImplementation.STACKED_KERNEL: _scaled_hankel_stacked_kernel,
}

DEFAULT_PFT_IMPLEMENTATION: PFTImplementation = PFTImplementation.STACKED_KERNEL
"""The implementation used when ``implementation`` is not given explicitly.

Hardcoded to the fastest implementation found by
``benchmarks/run_pft_benchmarks.py`` for the batched ``(radial, angular,
batch)`` scenario this default exists for: ``STACKED_KERNEL`` at ~4.7ms vs.
``HARMONIC_LOOP``'s ~5.2-6.2ms at ``n_angular=31, n_radial=128, batch=64``.
``HARMONIC_LOOP`` remains faster on a single, unbatched 2-D call (~2.7ms vs.
~3.6ms) -- a single batched ``matmul`` has nothing to amortize its own
per-call overhead against there -- but batching is exactly the scenario this
default is chosen for. No third, ``numba``-parallelized ``PARALLEL`` strategy
was added: ``STACKED_KERNEL``'s win here comes from a single BLAS call, not
from added parallelism, so the measurement that would justify one (being
Python-overhead-bound rather than BLAS-bound) never showed up. See
``pypft.dht``'s own ``DEFAULT_IMPLEMENTATION`` docstring for a related,
similarly-measured case (``VectorizedDHT``) that was kept despite losing on
every benchmark, rather than removed.
"""


def _validate_scaled_hankel_axes(
    values: np.ndarray, axis: int, angular_axis: int
) -> None:
    """Validate ``scaled_hankel``'s two axis arguments together.

    :param values: The array ``axis``/``angular_axis`` index into.
    :type values: np.ndarray
    :param axis: The radial axis.
    :type axis: int
    :param angular_axis: The angular (harmonic) axis.
    :type angular_axis: int
    :raises TypeError: If either axis argument is not an int.
    :raises ValueError: If either axis is out of bounds, or if they name the
        same axis.

    """
    IntValidator.type_is_int(value=axis)
    IntValidator.type_is_int(value=angular_axis)
    NumpyValidator.value_has_axis(value=values, axis=axis)
    NumpyValidator.value_has_axis(value=values, axis=angular_axis)
    if axis % values.ndim == angular_axis % values.ndim:
        raise ValueError(
            f"axis and angular_axis must name different axes of a "
            f"{values.ndim}-D value, both got {axis} and {angular_axis}"
        )


def scaled_hankel(
    values: np.ndarray,
    grid: PolarGrid,
    *,
    direction: Direction,
    axis: int,
    angular_axis: int,
    implementation: PFTImplementation = DEFAULT_PFT_IMPLEMENTATION,
) -> np.ndarray:
    """Apply ``grid``'s per-harmonic, sign- and scale-corrected Hankel transform.

    Loops over ``grid.harmonics`` -- one Hankel transform per harmonic, since
    each order needs its own kernel -- along ``axis``, applying
    ``hankel_transform``/``inverse_hankel_transform`` at that harmonic's
    ``abs(n)`` and folding in ``_harmonic_sign(n)`` plus the direction's own
    scale factor (forward: ``2*pi * i**(-n)``; inverse: ``i**n / (2*pi)`` --
    PeerJ CS Part II, Appendix A-5/A-6). ``implementation`` selects *how* that
    per-harmonic loop is carried out (``PFTImplementation``); which harmonic
    gets which order and scale factor is identical either way.

    This is the single place in ``src/`` that names the radial axis:
    ``forward_pft``/``inverse_pft`` always pass ``Axis.RADIAL``/
    ``Axis.ANGULAR`` explicitly rather than relying on a default, since
    ``hankel_transform`` itself defaults its own ``axis`` to ``-1`` for an
    unrelated, purely conventional reason (see ``pypft.axes``'s axis-default
    tiering).

    :param values: A 2-D array with one axis of length ``grid.n_radial`` (the
        radial axis, named by ``axis``) and the other of length
        ``grid.n_angular`` (named by ``angular_axis``), or a 3-D array adding
        one further batch axis on top of that.
    :type values: np.ndarray
    :param grid: The grid whose harmonics, order, and space limit drive the
        per-harmonic transform.
    :type grid: pypft.grid.PolarGrid
    :param direction: Whether to apply the forward or the inverse scaled
        Hankel transform.
    :type direction: Direction
    :param axis: The axis of ``values`` holding the radial samples.
    :type axis: int
    :param angular_axis: The axis of ``values`` holding the harmonic samples.
    :type angular_axis: int
    :param implementation: The strategy used to carry out the per-harmonic loop.
    :type implementation: PFTImplementation
    :returns: ``values`` with the scaled Hankel transform applied along
        ``axis``, one harmonic order at a time.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.

    """
    NumpyValidator.type_is_ndarray(value=values)
    NumpyValidator.value_is_2d_or_3d(value=values)
    _type_is_polar_grid(value=grid)
    EnumValidator.type_is_enum(value=direction)
    EnumValidator.value_is_enum_member(value=direction, enum_class=Direction)
    EnumValidator.type_is_enum(value=implementation)
    EnumValidator.value_is_enum_member(
        value=implementation, enum_class=PFTImplementation
    )
    _validate_scaled_hankel_axes(values=values, axis=axis, angular_axis=angular_axis)
    NumpyValidator.value1_axis_length_matches_value2(
        value1=values, axis1=angular_axis, value2=grid.harmonics, axis2=0
    )
    radial_reference = np.empty(grid.n_radial)
    NumpyValidator.value1_axis_length_matches_value2(
        value1=values, axis1=axis, value2=radial_reference, axis2=0
    )

    return _PFT_IMPLEMENTATIONS[implementation](
        values=values,
        grid=grid,
        direction=direction,
        axis=axis,
        angular_axis=angular_axis,
    )


# ======================================================================================
# The forward and inverse PFT
# ======================================================================================


def _validate_pft_input(values: np.ndarray, grid: PolarGrid, batch_axis: int) -> None:
    """Validate the shared arguments of ``forward_pft``/``inverse_pft``.

    :param values: The polar array to be transformed.
    :type values: np.ndarray
    :param grid: The sampling grid ``values`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``values`` holding the batch dimension, if
        any -- meaningful only when ``values`` is 3-D.
    :type batch_axis: int
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is not 2-D or 3-D, its radial/angular
        axes do not match ``grid``, ``batch_axis`` is given on a 2-D
        ``values``, or ``batch_axis`` does not name a 3-D ``values``'s last
        axis (PyPFT's own layout always places the batch axis last).

    """
    NumpyValidator.type_is_ndarray(value=values)
    NumpyValidator.value_is_2d_or_3d(value=values)
    _type_is_polar_grid(value=grid)
    IntValidator.type_is_int(value=batch_axis)
    if values.ndim == 2:
        if batch_axis != DEFAULT_BATCH_AXIS:
            raise ValueError(
                f"batch_axis={batch_axis} was given, but values is 2-D and has "
                "no batch axis"
            )
        reference = np.empty((grid.n_radial, grid.n_angular))
        NumpyValidator.value1_shape_matches_value2(value1=values, value2=reference)
    else:
        NumpyValidator.value_has_axis(value=values, axis=batch_axis)
        if batch_axis % values.ndim != Axis.BATCH:
            raise ValueError(
                f"batch_axis={batch_axis} does not name a 3-D value's last "
                "axis; PyPFT's own layout always places the batch axis last"
            )
        NumpyValidator.value1_axis_length_matches_value2(
            value1=values, axis1=Axis.RADIAL, value2=np.empty(grid.n_radial), axis2=0
        )
        NumpyValidator.value1_axis_length_matches_value2(
            value1=values,
            axis1=Axis.ANGULAR,
            value2=np.empty(grid.n_angular),
            axis2=0,
        )


def forward_pft(
    f: np.ndarray, grid: PolarGrid, *, batch_axis: int = DEFAULT_BATCH_AXIS
) -> np.ndarray:
    """Compute the forward polar Fourier transform (PFT).

    ``f(r, theta) --DFT_theta--> f_n(r) --H_n--> F_n(rho) --IDFT_phi-->
    F(rho, phi)``: an angular DFT turns the physical angle axis into a
    harmonic-order axis, ``scaled_hankel`` applies the order-specific radial
    transform, and an angular IDFT turns the harmonic axis back into a
    physical angle -- this time the frequency domain's own ``phi``.

    :param f: The space-domain samples ``f(r, theta)``, on ``grid``'s
        ``(n_radial, n_angular)`` layout (``pypft.axes.Axis``), or a 3-D
        ``(n_radial, n_angular, batch)`` stack of those.
    :type f: np.ndarray
    :param grid: The sampling grid ``f`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``f`` holding the batch dimension, only
        meaningful for a 3-D ``f`` -- PyPFT's own layout always places it
        last, so the only accepted value is ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :returns: The frequency-domain samples ``F(rho, phi)``, on the same layout.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``f`` is not 2-D or 3-D, or its shape does not
        match ``grid``.

    """
    _validate_pft_input(values=f, grid=grid, batch_axis=batch_axis)
    f_n = angular_dft(x=f, axis=Axis.ANGULAR)
    F_n = scaled_hankel(
        values=f_n,
        grid=grid,
        direction=Direction.FORWARD,
        axis=Axis.RADIAL,
        angular_axis=Axis.ANGULAR,
    )
    return inverse_angular_dft(X=F_n, axis=Axis.ANGULAR)


def inverse_pft(
    F: np.ndarray, grid: PolarGrid, *, batch_axis: int = DEFAULT_BATCH_AXIS
) -> np.ndarray:
    """Compute the inverse polar Fourier transform (IPFT).

    The exact mirror of ``forward_pft``: ``F(rho, phi) --DFT_phi--> F_n(rho)
    --H_n--> f_n(r) --IDFT_theta--> f(r, theta)``.

    :param F: The frequency-domain samples ``F(rho, phi)``, on ``grid``'s
        ``(n_radial, n_angular)`` layout (``pypft.axes.Axis``), or a 3-D
        ``(n_radial, n_angular, batch)`` stack of those.
    :type F: np.ndarray
    :param grid: The sampling grid ``F`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``F`` holding the batch dimension, only
        meaningful for a 3-D ``F`` -- PyPFT's own layout always places it
        last, so the only accepted value is ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :returns: The space-domain samples ``f(r, theta)``, on the same layout.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``F`` is not 2-D or 3-D, or its shape does not
        match ``grid``.

    """
    _validate_pft_input(values=F, grid=grid, batch_axis=batch_axis)
    F_n = angular_dft(x=F, axis=Axis.ANGULAR)
    f_n = scaled_hankel(
        values=F_n,
        grid=grid,
        direction=Direction.INVERSE,
        axis=Axis.RADIAL,
        angular_axis=Axis.ANGULAR,
    )
    return inverse_angular_dft(X=f_n, axis=Axis.ANGULAR)
