"""The ring-consistent route: an accurate, approximately invertible alternative path.

``pypft.forward_pft``/``pypft.inverse_pft`` (the exact path, and the default
everywhere) are Baddour's square, exactly invertible discrete transform. Their angular
DFT and inverse DFT each treat one radial index as a ring, which the transform's grid
never has: every spoke of ``pypft.grid.PolarGrid`` sits at its own radii, and the
transform identifies a spoke's angular sample index with a harmonic order (see
``DESIGN_NOTES.md``, "Grid: the spatial row index is a spoke, and the transform
identifies it with a harmonic"). As an approximation of the continuous 2-D Fourier
transform, that identification is the dominant error.

This module avoids it on both sides, without a new DHT or DFT kernel:

- **True rings in.** ``sample_harmonics_cartesian``/``sample_harmonics_uniform_polar``
  compute harmonic ``n`` from samples on true rings at its own radii ``r_nk`` (row
  ``i`` of ``PolarGrid.r`` for ``harmonics[i] == n``), by an angular quadrature around
  each ring. The result is a spatial-harmonic array: it enters the domain chain one
  step in, as a ``pypft.domains.PolarSpatialHarmonicSignal``'s values.
- **The existing per-harmonic DHT.** ``pypft.transform.scaled_hankel`` turns it into
  each harmonic's transform ``F_n`` at its own ``rho_nm``.
- **Each spoke's own radii out.** ``evaluate_frequency`` evaluates every ``F_n`` off
  its own grid, at every output spoke's ``rho_qm``, through the DHT's Fourier-Bessel
  expansion, and sums the harmonics on each spoke.

``forward_pft_ring`` composes the three, and ``inverse_pft_ring`` is its mirror image
(spokes interpolated onto true frequency rings, the inverse ``scaled_hankel``, and
``evaluate_space`` at each spoke's ``r_pk``). The route is accurate where the exact
path is not, but it is not a square matrix on one array, so it is not exactly
invertible: a round trip is an approximation, and the discrete rules of the exact path
(orthogonality, shift, convolution, Parseval) hold only approximately.
``DESIGN_NOTES.md``, "PFT: the exact path and the ring-consistent route," has the
measurements and the trade-off.

Every function takes and returns PyPFT's ``(radial, angular[, batch])`` layout
(``pypft.axes.Axis``), batch axis last, with a centered angular axis, and supports
``LimitKind.SPACE_LIMITED`` grids only.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import jv

from pypft.axes import DEFAULT_BATCH_AXIS, Axis
from pypft.dft import harmonics as _angular_harmonics
from pypft.dht._cached import CachedBesselDHT
from pypft.grid import (
    LimitKind,
    PolarGrid,
    _type_is_polar_grid,
    _uniform_polar_radii,
    _validate_uniform_polar,
)
from pypft.transform import Direction, _validate_pft_input, scaled_hankel
from pypft.utils.validators import IntValidator, NumpyValidator

#: The ranks ``sample_harmonics_cartesian`` accepts: a single ``(height, width)``
#: image, or a batch with a trailing batch axis.
_CARTESIAN_NDIMS = (2, 3)

#: One full turn, in radians.
_TURN = 2.0 * np.pi

# ======================================================================================
# Shared helpers
# ======================================================================================


def _require_space_limited(grid: PolarGrid, caller: str) -> None:
    """Reject a grid whose ``limit_kind`` this module does not support.

    :param grid: The grid to check.
    :type grid: pypft.grid.PolarGrid
    :param caller: The public function's name, for the error message.
    :type caller: str
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    if grid.limit_kind is not LimitKind.SPACE_LIMITED:
        raise NotImplementedError(
            f"{caller} only supports LimitKind.SPACE_LIMITED grids"
        )


def _orders(grid: PolarGrid) -> list[tuple[int, np.ndarray]]:
    """Group ``grid``'s harmonics by Bessel order.

    Harmonics ``n`` and ``-n`` share the order ``|n|``, hence the same rings, the
    same DHT kernel and the same off-grid Bessel matrices, so every per-order step
    below runs once per group rather than once per harmonic.

    :param grid: The grid whose harmonics to group.
    :type grid: pypft.grid.PolarGrid
    :returns: Every distinct order ``|n|``, ascending, with the angular indices of
        the harmonics of that order.
    :rtype: list[tuple[int, np.ndarray]]

    """
    magnitudes = np.abs(grid.harmonics)
    return [
        (int(order), np.flatnonzero(a=magnitudes == order))
        for order in np.unique(ar=magnitudes)
    ]


def _harmonic_coefficients(
    rings: np.ndarray, angles: np.ndarray, harmonics: np.ndarray, n_angular: int
) -> np.ndarray:
    """Compute the angular Fourier coefficients of ring samples, for given harmonics.

    The quadrature ``sum_s ring(phi_s) exp(-i n phi_s)`` over the ``len(angles)``
    samples of every ring, scaled by ``n_angular / len(angles)`` so the result is on
    ``pypft.dft.angular_dft``'s own scale: an ``n_angular``-point DFT of the same
    function returns these values.

    :param rings: The ring samples, ``(radial, len(angles)[, batch])``.
    :type rings: np.ndarray
    :param angles: The angle of every ring sample, in radians.
    :type angles: np.ndarray
    :param harmonics: The harmonic orders to compute.
    :type harmonics: np.ndarray
    :param n_angular: The angular sample count whose DFT scale the result is on.
    :type n_angular: int
    :returns: The coefficients, ``(radial, len(harmonics)[, batch])``.
    :rtype: np.ndarray

    """
    phases = np.exp(-1j * np.outer(a=angles, b=harmonics)) * (n_angular / len(angles))
    # tensordot contracts the ring axis and appends the harmonic axis last;
    # moving it back to position 1 restores the (radial, angular[, batch]) layout.
    coefficients = np.tensordot(a=rings, b=phases, axes=(Axis.ANGULAR, 0))
    return np.moveaxis(a=coefficients, source=-1, destination=Axis.ANGULAR)


def _off_grid_kernel(
    order_in: int, zeros_out: np.ndarray, zeros_in: np.ndarray, output: Direction
) -> np.ndarray:
    """Build the order-``order_in`` Fourier-Bessel matrix at another order's points.

    An order-``n`` DHT pair is a Fourier-Bessel expansion. With ``g = Y^{nN} x``,
    the dimensionless DHT of samples ``x`` at the order-``n`` points, the function
    the samples come from is ``(2 / j_nN) sum_k g_k J_n(u j_nk) / J_{n+1}(j_nk)**2``
    at the scaled radius ``u``: ``u = rho R / j_nN`` for a frequency-harmonic
    ``x`` (the forward DHT, ``rho r_nk = (rho R / j_nN) j_nk``), and ``u = r / R``
    for a spatial-harmonic ``x`` (the inverse DHT's Fourier-Bessel series,
    ``r rho_nk = (r / R) j_nk``). The output points are another order ``q``'s
    grid radii, ``rho_qm = j_qm / R`` or ``r_qm = j_qm R / j_qN``, so ``u`` is
    ``j_qm / j_nN`` or ``j_qm / j_qN``: the matrix is free of ``R`` either way. At
    ``q == n`` both reduce to ``Y^{nN}`` itself, so evaluating at the samples' own
    points reproduces them.

    The Bessel values come from one direct ``scipy.special.jv`` call, never an order
    recurrence; see ``DESIGN_NOTES.md``, "DHT: the kernel's Bessel values must be
    computed directly, never via order recurrence."

    :param order_in: The order ``n`` of the expansion.
    :type order_in: int
    :param zeros_out: The ``size + 1`` zeros of ``J_q``, the output points' order.
    :type zeros_out: np.ndarray
    :param zeros_in: The ``size + 1`` zeros of ``J_n``.
    :type zeros_in: np.ndarray
    :param output: ``Direction.FORWARD`` for frequency-domain output points (the
        values are frequency harmonics), ``Direction.INVERSE`` for space-domain ones
        (the values are spatial harmonics).
    :type output: pypft.transform.Direction
    :returns: The ``(size, size)`` matrix from ``g`` to the function at the output
        points.
    :rtype: np.ndarray

    """
    j_nN = zeros_in[-1]
    j_nk = zeros_in[:-1]
    denominator = j_nN if output is Direction.FORWARD else zeros_out[-1]
    argument = np.outer(a=zeros_out[:-1] / denominator, b=j_nk)
    return (2.0 / j_nN) * jv(order_in, argument) / jv(order_in + 1, j_nk) ** 2


def _evaluate_on_spokes(
    values: np.ndarray, grid: PolarGrid, *, output: Direction
) -> np.ndarray:
    """Evaluate per-harmonic values at every spoke's own radii, then sum per spoke.

    Column ``i`` of ``values`` is harmonic ``n = grid.harmonics[i]``, sampled at its
    own order-``|n|`` points. Every harmonic is evaluated at every spoke's radii
    through ``_off_grid_kernel``, and spoke ``q`` is the inverse angular DFT of those
    values at its own angle, ``(1 / n_angular) sum_n F_n e^{i n psi_q}``.

    The ``n_angular**2`` off-grid evaluations, ``n_radial**2`` Bessel values each,
    dominate the cost. ``scipy.special.jv`` releases the GIL, so the output orders
    are evaluated on a thread pool, each writing its own spokes; no matrix is
    cached, since one per pair of orders would need ``(n_angular / 2)**2`` matrices
    of ``n_radial**2`` values each.

    :param values: Per-harmonic values, ``(radial, angular[, batch])``.
    :type values: np.ndarray
    :param grid: The grid whose harmonics, orders and spokes drive the evaluation.
    :type grid: pypft.grid.PolarGrid
    :param output: ``Direction.FORWARD`` to evaluate frequency harmonics at the
        frequency spokes' radii, ``Direction.INVERSE`` for spatial harmonics at the
        spatial spokes' radii.
    :type output: pypft.transform.Direction
    :returns: The values at every spoke's own radii, on the same layout.
    :rtype: np.ndarray

    """
    size = grid.n_radial
    orders = _orders(grid=grid)
    harmonics = grid.harmonics
    zeros: dict[int, np.ndarray] = {}
    # The DHT of every harmonic, along the radial axis: the expansion coefficients
    # every off-grid evaluation below starts from.
    expansion = np.empty(values.shape, dtype=complex)
    for order, columns in orders:
        kernel, zeros[order] = CachedBesselDHT._bessel_kernel(n=order, size=size)
        expansion[:, columns, ...] = np.tensordot(
            a=kernel, b=values[:, columns, ...], axes=(1, Axis.RADIAL)
        )

    result = np.empty(values.shape, dtype=complex)

    def evaluate_order(order_out: int, spokes: np.ndarray) -> None:
        """Fill the spokes of one order: every harmonic at their radii, summed.

        :param order_out: The order of the spokes' radii.
        :type order_out: int
        :param spokes: The angular indices of the spokes of that order.
        :type spokes: np.ndarray

        """
        total = np.zeros((size, len(spokes)) + values.shape[2:], dtype=complex)
        for order_in, columns in orders:
            off_grid = _off_grid_kernel(
                order_in=order_in,
                zeros_out=zeros[order_out],
                zeros_in=zeros[order_in],
                output=output,
            )
            # Every harmonic of order_in at the spokes' radii, (radial, k[, batch]).
            at_radii = np.tensordot(
                a=off_grid, b=expansion[:, columns, ...], axes=(1, Axis.RADIAL)
            )
            # The inverse angular DFT's own terms, e^{i n psi_q} / n_angular, for
            # these harmonics (rows) at these spokes (columns).
            phases = (
                np.exp(1j * np.outer(a=harmonics[columns], b=grid.psi[spokes]))
                / grid.n_angular
            )
            summed = np.tensordot(a=at_radii, b=phases, axes=(Axis.ANGULAR, 0))
            total += np.moveaxis(a=summed, source=-1, destination=Axis.ANGULAR)
        result[:, spokes, ...] = total

    with ThreadPoolExecutor() as pool:
        # list() drains the iterator so an exception in any worker is re-raised.
        list(pool.map(lambda item: evaluate_order(*item), orders))
    return result


# ======================================================================================
# True rings in
# ======================================================================================


def _default_n_quadrature(grid: PolarGrid) -> int:
    """Choose the angular quadrature size for sampling a Cartesian image.

    One sample per pixel of arc on the outermost ring (``2 * pi * R``, with ``R`` in
    pixels, as ``pypft.grid.sample_cartesian`` reads it), and never fewer than the
    grid's own spokes.

    :param grid: The grid to sample on.
    :type grid: pypft.grid.PolarGrid
    :returns: The number of angles sampled around every ring.
    :rtype: int

    """
    return max(grid.n_angular, int(np.ceil(_TURN * grid.R)))


def _remap_rings(plane: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample one real image plane at given pixel coordinates, bilinearly.

    :param plane: A real ``(height, width)`` image.
    :type plane: np.ndarray
    :param map_x: The column coordinate of every sample.
    :type map_x: np.ndarray
    :param map_y: The row coordinate of every sample.
    :type map_y: np.ndarray
    :returns: The image at every sample, of ``map_x``'s shape.
    :rtype: np.ndarray

    """
    return cv2.remap(
        src=np.ascontiguousarray(a=plane, dtype=np.float64),
        map1=map_x,
        map2=map_y,
        interpolation=cv2.INTER_LINEAR,
    )


def sample_harmonics_cartesian(
    image: np.ndarray, grid: PolarGrid, *, n_quadrature: int | None = None
) -> np.ndarray:
    """Compute every harmonic of a Cartesian image on its own true rings.

    Harmonic ``n = grid.harmonics[i]`` is sampled on rings at its own radii
    ``r_nk`` (row ``i`` of ``grid.r``): the image is evaluated at ``n_quadrature``
    equally spaced angles around each ring, bilinearly, in
    ``pypft.grid.sample_cartesian``'s own convention (center at
    ``(width / 2, height / 2)``, radii in pixels, angles measured on image
    coordinates with no ``y``-flip), and
    the angular quadrature gives each harmonic's coefficient on
    ``pypft.dft.angular_dft``'s scale. The result is the ring-consistent counterpart
    of ``angular_dft(sample_cartesian(image, grid).T)``: a spatial-harmonic array,
    the values of a ``pypft.domains.PolarSpatialHarmonicSignal``.

    An image offers any number of angles, so the quadrature is not limited to the
    grid's spokes; more angles reduce the aliasing of harmonics beyond
    ``grid.n_angular``'s range into the ones computed.

    :param image: A real or complex ``(height, width)`` image, or a batch
        ``(height, width, batch)``.
    :type image: np.ndarray
    :param grid: The space-limited grid whose harmonics and rings to sample.
    :type grid: pypft.grid.PolarGrid
    :param n_quadrature: The number of angles sampled around every ring; by
        default one per pixel of arc on the outermost ring, ``ceil(2 * pi *
        grid.R)``, and at least ``grid.n_angular``.
    :type n_quadrature: int | None
    :returns: The spatial-harmonic array, ``(grid.n_radial, grid.n_angular[,
        batch])``, complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``image`` is not 2-D or 3-D or has a non-finite element,
        or ``n_quadrature`` is not strictly positive.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    NumpyValidator.type_is_ndarray(value=image)
    NumpyValidator.value_has_ndim_in(value=image, ndims=_CARTESIAN_NDIMS)
    NumpyValidator.value_is_finite(value=image)
    _type_is_polar_grid(value=grid)
    _require_space_limited(grid=grid, caller="sample_harmonics_cartesian")
    if n_quadrature is None:
        n_quadrature = _default_n_quadrature(grid=grid)
    else:
        IntValidator.type_is_int(value=n_quadrature)
        IntValidator.value_is_positive(value=n_quadrature)

    height, width = image.shape[0], image.shape[1]
    center_x, center_y = width / 2.0, height / 2.0
    angles = np.arange(n_quadrature) * (_TURN / n_quadrature)
    # A single image is a batch of one, so one loop serves both ranks.
    planes = image[..., np.newaxis] if image.ndim == 2 else image
    result = np.empty((grid.n_radial, grid.n_angular, planes.shape[-1]), dtype=complex)
    radii = grid.r
    for _, columns in _orders(grid=grid):
        # The rings of harmonics n and -n coincide (rows of equal order).
        ring_radii = radii[columns[0]][:, np.newaxis]
        map_x = (center_x + ring_radii * np.cos(angles)).astype(np.float32)
        map_y = (center_y + ring_radii * np.sin(angles)).astype(np.float32)
        for element in range(planes.shape[-1]):
            plane = planes[..., element]
            rings = _remap_rings(plane=plane.real, map_x=map_x, map_y=map_y)
            if np.iscomplexobj(plane):
                # The interpolation is linear, so the two parts are sampled apart.
                imaginary = _remap_rings(plane=plane.imag, map_x=map_x, map_y=map_y)
                rings = rings + 1j * imaginary
            result[:, columns, element] = _harmonic_coefficients(
                rings=rings,
                angles=angles,
                harmonics=grid.harmonics[columns],
                n_angular=grid.n_angular,
            )
    return result[..., 0] if image.ndim == 2 else result


def sample_harmonics_uniform_polar(
    values: np.ndarray, grid: PolarGrid, *, radius: float
) -> np.ndarray:
    """Compute every harmonic of uniform polar data on its own true rings.

    ``values`` is uniform polar data in ``pypft.grid.resample_uniform_polar``'s exact
    convention: radial sample ``k`` of ``n`` at ``k * radius / n`` on every spoke,
    and uniform spokes on a centered angular axis. Equal radii on every spoke make
    each radial index a true ring already, so only the radius has to be
    interpolated: every spoke is interpolated along the radius with a cubic spline
    onto each harmonic's own radii ``r_nk`` (row ``i`` of ``grid.r`` for
    ``grid.harmonics[i] == n``), and the angular quadrature over the data's own
    spokes gives each harmonic's coefficient on ``pypft.dft.angular_dft``'s scale.
    No angle is interpolated, and no sample is needed beyond the data's own spokes.

    The data's spokes are the quadrature, so a harmonic outside their own centered
    range aliases onto one inside it; fewer spokes than ``grid.n_angular`` warn.

    :param values: A uniform polar array, ``(radial, angular)`` or a batch
        ``(radial, angular, batch)``, real or complex, with at least two radial
        samples and a centered angular axis.
    :type values: np.ndarray
    :param grid: The space-limited grid whose harmonics and rings to sample.
    :type grid: pypft.grid.PolarGrid
    :param radius: The radius the uniform samples cover, in the same unit as
        ``grid.R``.
    :type radius: float
    :returns: The spatial-harmonic array, ``(grid.n_radial, grid.n_angular[,
        batch])``, complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is not 2-D or 3-D, has fewer than two radial
        samples, no spokes, or a non-finite element, or ``radius`` is not strictly
        positive.
    :raises pypft.grid.RadiusCoverageError: If ``grid.R`` exceeds ``radius``.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    return _sample_harmonics_uniform_polar(
        values=values,
        grid=grid,
        radius=radius,
        caller="sample_harmonics_uniform_polar",
    )


def _sample_harmonics_uniform_polar(
    values: np.ndarray, grid: PolarGrid, *, radius: float, caller: str
) -> np.ndarray:
    """Validate and sample uniform polar data; see ``sample_harmonics_uniform_polar``.

    Every public entry point calls this directly, so the angular-upsampling warning
    reaches the user's call at one fixed stack depth.

    :param values: The uniform polar array.
    :type values: np.ndarray
    :param grid: The grid whose harmonics and rings to sample.
    :type grid: pypft.grid.PolarGrid
    :param radius: The radius the uniform samples cover.
    :type radius: float
    :param caller: The public function's name, for the error messages.
    :type caller: str
    :returns: The spatial-harmonic array.
    :rtype: np.ndarray

    """
    # stacklevel=4: past the validator, this helper and the public entry point.
    _validate_uniform_polar(
        values=values, grid=grid, radius=radius, caller=caller, stacklevel=4
    )
    n_radial, n_spokes = values.shape[Axis.RADIAL], values.shape[Axis.ANGULAR]
    uniform_radii = _uniform_polar_radii(n_radial=n_radial, radius=radius)
    # The data's own spokes, in the same centered order as every stored angular axis.
    angles = _angular_harmonics(n_spokes) * (_TURN / n_spokes)
    # One radial spline through every spoke (and batch element) at once.
    spline = CubicSpline(x=uniform_radii, y=values, axis=Axis.RADIAL)

    result = np.empty((grid.n_radial, grid.n_angular) + values.shape[2:], dtype=complex)
    radii = grid.r
    for _, columns in _orders(grid=grid):
        # Every spoke at this order's radii: true rings, (radial, spokes[, batch]).
        rings = spline(radii[columns[0]])
        result[:, columns, ...] = _harmonic_coefficients(
            rings=rings,
            angles=angles,
            harmonics=grid.harmonics[columns],
            n_angular=grid.n_angular,
        )
    return result


def _harmonics_from_spokes(values: np.ndarray, grid: PolarGrid) -> np.ndarray:
    """Compute every harmonic of frequency-domain grid samples on true rings.

    The mirror image of ``sample_harmonics_uniform_polar``, for samples on the
    grid's own frequency spokes: spoke ``q`` sits at its own radii ``rho_qm``, so
    every spoke is interpolated along the radius with a cubic spline onto each
    harmonic's radii ``rho_nm``, and the quadrature over the ``n_angular`` spokes
    gives the coefficients.

    A ring's radius can fall outside a spoke's own samples, by up to about
    ``(|n| - |q|) pi / (2 R)`` past its last one and, inside the central gap, below
    its first one (``rho_q1`` grows with ``|q|``). Past the last sample the value is
    taken as ``0``, the grid's own band-limit assumption, since extrapolating a
    cubic over several sample spacings diverges; inside the gap the spline's first
    piece is extrapolated, which is accurate only while the gap holds little of the
    transform. ``DESIGN_NOTES.md``, "PFT: the exact path and the ring-consistent
    route," has the measurements.

    :param values: Frequency-domain samples on ``grid``, ``(radial, angular[,
        batch])``.
    :type values: np.ndarray
    :param grid: The grid the samples are on.
    :type grid: pypft.grid.PolarGrid
    :returns: The frequency-harmonic array, on the same layout.
    :rtype: np.ndarray

    """
    radii = grid.rho
    splines = [
        CubicSpline(x=radii[spoke], y=values[:, spoke, ...], axis=Axis.RADIAL)
        for spoke in range(grid.n_angular)
    ]
    result = np.empty(values.shape, dtype=complex)
    # Broadcasts a (radial,) mask against a (radial[, batch]) spoke.
    trailing = (1,) * (values.ndim - 2)
    for _, columns in _orders(grid=grid):
        ring_radii = radii[columns[0]]
        rings = np.stack(
            arrays=[
                np.where(
                    (ring_radii > radii[spoke, -1]).reshape((-1,) + trailing),
                    0.0,
                    spline(ring_radii),
                )
                for spoke, spline in enumerate(splines)
            ],
            axis=Axis.ANGULAR,
        )
        result[:, columns, ...] = _harmonic_coefficients(
            rings=rings,
            angles=grid.psi,
            harmonics=grid.harmonics[columns],
            n_angular=grid.n_angular,
        )
    return result


# ======================================================================================
# Each spoke's own radii out
# ======================================================================================


def evaluate_frequency(
    harmonics: np.ndarray, grid: PolarGrid, *, batch_axis: int = DEFAULT_BATCH_AXIS
) -> np.ndarray:
    """Evaluate frequency-harmonic values at every spoke's own radii.

    Column ``i`` of ``harmonics`` is harmonic ``n = grid.harmonics[i]``'s transform
    ``F_n`` at its own ``rho_nm`` (row ``i`` of ``grid.rho``), as
    ``pypft.transform.scaled_hankel`` or a
    ``pypft.domains.PolarFrequencyHarmonicSignal`` holds it. Every ``F_n`` is
    evaluated off its own points, at every spoke's ``rho_qm``, through the
    order-``|n|`` DHT's Fourier-Bessel expansion; then spoke ``q`` is
    ``(1 / n_angular) sum_n F_n(rho_qm) e^{i n psi_q}``, the inverse angular DFT at
    its own radii. The output is on ``pypft.forward_pft``'s frequency grid
    (``grid.rho.T``, ``grid.psi``), without the spoke-to-harmonic identification of
    ``pypft.dft.inverse_angular_dft``.

    The cost is ``n_angular**2 / 4`` Bessel matrices of ``n_radial**2`` values
    each, computed on a thread pool.

    :param harmonics: Frequency-harmonic values, ``(n_radial, n_angular)`` or a batch
        ``(n_radial, n_angular, batch)``.
    :type harmonics: np.ndarray
    :param grid: The space-limited grid the values are on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``harmonics`` holding the batch dimension; PyPFT's
        own layout always places it last, so the only accepted value is
        ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :returns: The frequency-domain samples at ``(rho_qm, psi_q)``, on the same
        layout, complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``harmonics`` is not 2-D or 3-D, or its shape does not
        match ``grid``.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    _validate_pft_input(values=harmonics, grid=grid, batch_axis=batch_axis)
    _require_space_limited(grid=grid, caller="evaluate_frequency")
    return _evaluate_on_spokes(values=harmonics, grid=grid, output=Direction.FORWARD)


def evaluate_space(
    harmonics: np.ndarray, grid: PolarGrid, *, batch_axis: int = DEFAULT_BATCH_AXIS
) -> np.ndarray:
    """Evaluate spatial-harmonic values at every spoke's own radii.

    The space-side mirror of ``evaluate_frequency``: column ``i`` of ``harmonics``
    is harmonic ``n = grid.harmonics[i]`` at its own ``r_nk`` (row ``i`` of
    ``grid.r``), as ``sample_harmonics_cartesian`` returns it or a
    ``pypft.domains.PolarSpatialHarmonicSignal`` holds it. Each harmonic is evaluated
    at every spoke's ``r_pk`` through the inverse DHT's Fourier-Bessel series,
    ``f_n(r) = (2 / R**2) sum_m F_n(rho_nm) J_|n|(r rho_nm) / J_{|n|+1}(j_|n|m)**2``,
    and spoke ``p`` is the inverse angular DFT at its own radii. The output is on
    ``pypft.inverse_pft``'s spatial grid (``grid.r.T``, ``grid.theta``).

    :param harmonics: Spatial-harmonic values, ``(n_radial, n_angular)`` or a batch
        ``(n_radial, n_angular, batch)``.
    :type harmonics: np.ndarray
    :param grid: The space-limited grid the values are on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``harmonics`` holding the batch dimension; PyPFT's
        own layout always places it last, so the only accepted value is
        ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :returns: The space-domain samples at ``(r_pk, theta_p)``, on the same layout,
        complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``harmonics`` is not 2-D or 3-D, or its shape does not
        match ``grid``.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    _validate_pft_input(values=harmonics, grid=grid, batch_axis=batch_axis)
    _require_space_limited(grid=grid, caller="evaluate_space")
    return _evaluate_on_spokes(values=harmonics, grid=grid, output=Direction.INVERSE)


# ======================================================================================
# The forward and inverse ring-consistent transforms
# ======================================================================================


def forward_pft_ring(
    values: np.ndarray, grid: PolarGrid, *, radius: float | None = None
) -> np.ndarray:
    """Compute the forward PFT by the ring-consistent route.

    ``sample_harmonics_cartesian`` (``radius`` omitted: ``values`` is a Cartesian
    image) or ``sample_harmonics_uniform_polar`` (``radius`` given: ``values`` is
    uniform polar data covering it), then the forward
    ``pypft.transform.scaled_hankel``, then ``evaluate_frequency``. The result is on
    ``pypft.forward_pft``'s own frequency grid (``grid.rho.T``, ``grid.psi``), and
    approximates the continuous 2-D Fourier transform there without the
    spoke-to-harmonic identification error of the exact path. It is not exactly
    invertible; ``inverse_pft_ring`` is an approximate inverse.

    For control over the input stage (e.g. ``sample_harmonics_cartesian``'s
    ``n_quadrature``), call a sampler directly, step its result along the domain
    chain with ``pypft.domains.PolarSpatialHarmonicSignal``'s
    ``to_polar_frequency_harmonic``, and pass that signal's values to
    ``evaluate_frequency``.

    :param values: A Cartesian image ``(height, width[, batch])`` when ``radius`` is
        omitted, else a uniform polar array ``(radial, angular[, batch])``; see the
        two samplers.
    :type values: np.ndarray
    :param grid: The space-limited grid to transform on.
    :type grid: pypft.grid.PolarGrid
    :param radius: The radius uniform polar ``values`` cover, in the same unit as
        ``grid.R``; omitted for a Cartesian image.
    :type radius: float | None
    :returns: The frequency-domain samples at ``(rho_qm, psi_q)``, ``(grid.n_radial,
        grid.n_angular[, batch])``, complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.
    :raises pypft.grid.RadiusCoverageError: If ``radius`` is given and ``grid.R``
        exceeds it.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    if radius is None:
        f_n = sample_harmonics_cartesian(image=values, grid=grid)
    else:
        f_n = _sample_harmonics_uniform_polar(
            values=values, grid=grid, radius=radius, caller="forward_pft_ring"
        )
    F_n = scaled_hankel(
        values=f_n,
        grid=grid,
        direction=Direction.FORWARD,
        axis=Axis.RADIAL,
        angular_axis=Axis.ANGULAR,
    )
    return _evaluate_on_spokes(values=F_n, grid=grid, output=Direction.FORWARD)


def inverse_pft_ring(
    F: np.ndarray, grid: PolarGrid, *, batch_axis: int = DEFAULT_BATCH_AXIS
) -> np.ndarray:
    """Compute the inverse PFT by the ring-consistent route: an approximation.

    The mirror image of ``forward_pft_ring``, from samples on ``pypft.inverse_pft``'s
    own frequency grid (``grid.rho.T``, ``grid.psi``): every frequency spoke is
    interpolated along the radius onto each harmonic's own true rings ``rho_nm``
    (a cubic spline per spoke), the angular quadrature over the spokes gives each
    harmonic, the inverse ``pypft.transform.scaled_hankel`` transforms it, and
    ``evaluate_space`` evaluates the result at every spatial spoke's own ``r_pk``.

    This is **not** an exact inverse of ``forward_pft_ring``: neither direction is a
    square matrix on one array, and the input stage interpolates. The result
    approximates the continuous inverse transform at ``(r_pk, theta_p)``, and a round
    trip returns the input only approximately; use ``pypft.inverse_pft`` for an exact
    inverse. Its accuracy is limited by the frequency grid's central gap: spoke ``q``
    has no sample below ``rho_q1 = j_|q|1 / R``, so the input stage extrapolates
    there, which is accurate only when ``R`` is large relative to the object and
    ``n_angular`` small (a round trip within ``1.4e-4`` relative L2 at
    ``n_radial=382, n_angular=15, R=40``), and fails when the gap holds much of the
    transform (``1.6`` at ``n_radial=200, n_angular=63, R=10``). ``DESIGN_NOTES.md``,
    "PFT: the exact path and the ring-consistent route," has the measurements.

    :param F: Frequency-domain samples on ``grid``, ``(n_radial, n_angular)`` or a
        batch ``(n_radial, n_angular, batch)``.
    :type F: np.ndarray
    :param grid: The space-limited grid the samples are on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``F`` holding the batch dimension; PyPFT's own
        layout always places it last, so the only accepted value is
        ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :returns: The space-domain samples at ``(r_pk, theta_p)``, on the same layout,
        complex.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``F`` is not 2-D or 3-D, has a non-finite element, or its
        shape does not match ``grid``.
    :raises NotImplementedError: If ``grid.limit_kind`` is not
        ``LimitKind.SPACE_LIMITED``.

    """
    _validate_pft_input(values=F, grid=grid, batch_axis=batch_axis)
    NumpyValidator.value_is_finite(value=F)
    _require_space_limited(grid=grid, caller="inverse_pft_ring")
    F_n = _harmonics_from_spokes(values=F, grid=grid)
    f_n = scaled_hankel(
        values=F_n,
        grid=grid,
        direction=Direction.INVERSE,
        axis=Axis.RADIAL,
        angular_axis=Axis.ANGULAR,
    )
    return _evaluate_on_spokes(values=f_n, grid=grid, output=Direction.INVERSE)
