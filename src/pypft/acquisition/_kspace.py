"""Radially acquired k-space, converted to PyPFT's uniform polar layout.

A radial acquisition records k-space along *lines*: each line passes through the
center of k-space at some angle, and its readout samples are equally spaced along it,
on both sides of the center. PyPFT stores polar data as *spokes* instead, half-lines
starting at the center (``pypft.axes.PolarAxis``): ``(radial, angular[, batch])``, with
a centered angular axis. ``radial_lines_to_polar`` splits every line at its center
sample into two spokes, the samples after the center at the line's own angle and the
samples before it, reversed, at the opposite angle. ``L`` lines at uniform angles over
half a turn become ``2 * L`` uniform spokes over a full turn: a uniform polar array in
the frequency domain, ``pypft.grid.resample_uniform_polar_frequency``'s and
``pypft.inverse_pft_ring``'s input. ``RadialKSpace`` carries it, with the frequency
radius it covers.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pypft.axes import _value_is_polar_sample_or_batch
from pypft.utils.validators import (
    FloatValidator,
    IntValidator,
    NumpyValidator,
    PathValidator,
)

#: One full turn, in radians.
_TURN = 2.0 * np.pi

#: The ranks of a lines array: ``(readout, line)``, or a batch with a trailing axis.
_LINES_NDIMS = (2, 3)

#: The readout axis of a lines array, the axis of the samples along each line.
_READOUT_AXIS = 0

#: The line axis of a lines array.
_LINE_AXIS = 1

#: How far, in units of the uniform angular step, a line's angle may lie from the
#: nearest multiple of that step and still count as on it.
_ANGLE_TOLERANCE = 1e-6

#: The file suffix ``RadialKSpace.save``/``RadialKSpace.load`` use.
_KSPACE_SUFFIX = ".npz"

#: The keys of the arrays a saved ``RadialKSpace`` holds.
_VALUES_KEY, _RADIUS_KEY = "values", "radius"


class NonUniformAnglesError(ValueError):
    """Raised when radial lines are not at uniform angles over half a turn."""


# ======================================================================================
# The converted data
# ======================================================================================


@dataclass(frozen=True, eq=False)
class RadialKSpace:
    """Radially acquired k-space in PyPFT's uniform polar layout.

    ``values`` is ``(radial, angular)`` or a batch ``(radial, angular, batch)``, with a
    centered angular axis: spoke ``i`` at angle ``(i - n_angular // 2) * 2 * pi /
    n_angular``, in PyPFT's angle convention (measured on image coordinates, from
    ``+x`` toward ``+y``). Radial sample ``k`` of ``n_radial`` sits at frequency
    ``k * radius / n_radial`` on every spoke, so ``radius`` is the ``radius`` argument
    of ``pypft.grid.resample_uniform_polar_frequency`` and of
    ``pypft.inverse_pft_ring``. It is in radians per pixel, where a pixel is one
    sample of the image an inverse DFT of one whole readout gives: a grid's ``R`` in
    those pixels matches it.

    Not comparable with ``==`` (``eq=False``), since its ``values`` is an array.

    :param values: The uniform polar frequency-domain samples, real or complex.
    :type values: np.ndarray
    :param radius: The frequency radius the samples cover, in radians per pixel.
    :type radius: float
    :raises TypeError: If any field has the wrong type.
    :raises ValueError: If ``values`` is neither a single sample nor a batch or has a
        non-finite element, or ``radius`` is not strictly positive.

    """

    values: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        """Validate every field once, right after construction."""
        NumpyValidator.type_is_ndarray(value=self.values)
        _value_is_polar_sample_or_batch(value=self.values)
        NumpyValidator.value_is_finite(value=self.values)
        FloatValidator.type_is_float(value=self.radius)
        FloatValidator.value_is_positive(value=self.radius)

    def save(self, path: Path) -> None:
        """Write the data to an uncompressed NumPy ``.npz`` archive.

        The archive holds two arrays, ``values`` and ``radius``; ``RadialKSpace.load``
        reads it back, and so does ``numpy.load`` directly.

        :param path: The file to write, with a ``.npz`` suffix; an existing file is
            replaced.
        :type path: Path
        :raises TypeError: If ``path`` is not a ``pathlib.Path``.
        :raises ValueError: If ``path`` does not end in ``.npz``.
        :raises OSError: If ``path`` cannot be written.

        """
        PathValidator.type_is_Path(value=path)
        PathValidator.value_has_correct_suffix(value=path, suffix=_KSPACE_SUFFIX)
        PathValidator.value_is_writable(value=path)
        np.savez(
            file=path,
            **{_VALUES_KEY: self.values, _RADIUS_KEY: np.float64(self.radius)},
        )

    @classmethod
    def load(cls, path: Path) -> "RadialKSpace":
        """Read data written by ``RadialKSpace.save``.

        :param path: The ``.npz`` file to read.
        :type path: Path
        :returns: The data, validated as on construction.
        :rtype: RadialKSpace
        :raises TypeError: If ``path`` is not a ``pathlib.Path``.
        :raises ValueError: If ``path`` does not end in ``.npz``, lacks the
            ``values``/``radius`` arrays, or holds invalid data.
        :raises OSError: If ``path`` does not exist, is a directory, or cannot be
            read.

        """
        PathValidator.type_is_Path(value=path)
        PathValidator.value_has_correct_suffix(value=path, suffix=_KSPACE_SUFFIX)
        PathValidator.value_is_a_file(value=path)
        PathValidator.value_is_readable(value=path)
        # allow_pickle=False: an archive is read as plain arrays, never as code.
        with np.load(file=path, allow_pickle=False) as archive:
            missing = {_VALUES_KEY, _RADIUS_KEY} - set(archive.files)
            if missing:
                raise ValueError(
                    f"{path} is not a saved RadialKSpace: it lacks {sorted(missing)}"
                )
            values = archive[_VALUES_KEY]
            radius = float(archive[_RADIUS_KEY])
        return cls(values=values, radius=radius)


# ======================================================================================
# Lines to spokes
# ======================================================================================


def _angular_slots(angles: np.ndarray) -> np.ndarray:
    """Find each line's slot among ``len(angles)`` uniform angles over half a turn.

    :param angles: The angle of every line, in radians.
    :type angles: np.ndarray
    :returns: Each line's slot ``m``, its angle being ``m * pi / len(angles)``
        modulo a full turn, as an integer in ``0 .. 2 * len(angles) - 1``.
    :rtype: np.ndarray
    :raises NonUniformAnglesError: If an angle is not a multiple of the uniform step,
        or two lines share a slot modulo half a turn.

    """
    n_lines = len(angles)
    steps = angles / (np.pi / n_lines)
    slots = np.rint(steps)
    if not np.allclose(a=steps, b=slots, rtol=0.0, atol=_ANGLE_TOLERANCE):
        raise NonUniformAnglesError(
            f"every line angle must be a multiple of pi / {n_lines}, the uniform step "
            f"of {n_lines} lines over half a turn"
        )
    # Line angles are only defined modulo a full turn; 2 * n_lines slots fill it.
    slots = np.mod(slots.astype(int), 2 * n_lines)
    # A line covers its slot and the opposite one, so lines must differ modulo half
    # a turn for the 2 * n_lines spokes to fill the turn exactly once.
    if len(np.unique(ar=np.mod(slots, n_lines))) != n_lines:
        raise NonUniformAnglesError(
            f"the {n_lines} lines must cover {n_lines} distinct angles modulo pi, "
            "each a multiple of the uniform step"
        )
    return slots


def radial_lines_to_polar(
    lines: np.ndarray, *, angles: np.ndarray, center_index: int
) -> RadialKSpace:
    """Split radial k-space lines at their center into PyPFT's uniform polar spokes.

    ``lines`` is ``(readout, line)`` or a batch ``(readout, line, batch)``: readout
    sample ``s`` of line ``l`` sits at k-space position ``(s - center_index) * dk`` in
    the direction ``angles[l]``, with ``dk`` the readout's own sample spacing. Every
    line becomes two spokes: samples ``center_index, center_index + 1, ...`` at
    ``angles[l]``, and samples ``center_index, center_index - 1, ...`` at
    ``angles[l] + pi``. Both spokes keep ``min(center_index + 1, readout -
    center_index)`` samples, the longer side being cut to the shorter.

    The lines must be at uniform angles over half a turn, ``m * pi / L`` for ``L``
    lines, each multiple once modulo ``pi``, in any order (sequential, interleaved,
    reversed); their ``2 * L`` spokes are then exactly the uniform, centered spokes of
    a ``2 * L``-spoke array. Angles are in PyPFT's convention: measured on image
    coordinates, from ``+x`` toward ``+y``. Golden-angle and other non-uniform
    trajectories have no uniform polar layout and are rejected.

    A pixel being one sample of the image an inverse DFT of one whole readout gives,
    ``dk`` is ``2 * pi / readout`` radians per pixel, and the returned ``radius`` is
    ``n_radial * 2 * pi / readout``.

    :param lines: The k-space lines, real or complex, ``(readout, line[, batch])``.
    :type lines: np.ndarray
    :param angles: The angle of every line's samples past its center, in radians,
        ``(line,)``.
    :type angles: np.ndarray
    :param center_index: The readout index of the center of k-space.
    :type center_index: int
    :returns: The spokes, ``(n_radial, 2 * L[, batch])`` with a centered angular
        axis, and the frequency radius they cover.
    :rtype: RadialKSpace
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``lines`` is neither ``(readout, line)`` nor a batch, or has
        a non-finite element; ``angles`` is not one finite angle per line; or
        ``center_index`` is outside the readout.
    :raises NonUniformAnglesError: If the lines are not at uniform angles over half a
        turn.

    """
    NumpyValidator.type_is_ndarray(value=lines)
    NumpyValidator.value_has_ndim_in(value=lines, ndims=_LINES_NDIMS)
    NumpyValidator.value_is_finite(value=lines)
    NumpyValidator.type_is_ndarray(value=angles)
    NumpyValidator.value_is_1d(value=angles)
    NumpyValidator.value_is_finite(value=angles)
    NumpyValidator.value1_axis_length_matches_value2(
        value1=lines, axis1=_LINE_AXIS, value2=angles, axis2=0
    )
    IntValidator.type_is_int(value=center_index)
    IntValidator.value_is_non_negative(value=center_index)
    n_readout, n_lines = lines.shape[_READOUT_AXIS], lines.shape[_LINE_AXIS]
    IntValidator.value1_is_less_than_value2(value1=center_index, value2=n_readout)
    slots = _angular_slots(angles=angles)

    n_radial = min(center_index + 1, n_readout - center_index)
    # Both halves of every line, starting at the center sample: the forward half
    # along the readout, the backward half reversed.
    forward = lines[slice(center_index, center_index + n_radial)]
    backward = lines[center_index::-1][:n_radial]
    values = np.empty((n_radial, 2 * n_lines) + lines.shape[2:], dtype=lines.dtype)
    # A spoke at slot m (angle m * pi / L) sits at centered index (m + L) mod 2L,
    # since centered index i holds angle (i - L) * pi / L; the backward half is one
    # half turn, L slots, further round.
    values[:, np.mod(slots + n_lines, 2 * n_lines), ...] = forward
    values[:, slots, ...] = backward
    radius = n_radial * _TURN / n_readout
    return RadialKSpace(values=values, radius=radius)
