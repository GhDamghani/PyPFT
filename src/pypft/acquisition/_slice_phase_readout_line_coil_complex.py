"""Reader for the ``SLICE_PHASE_READOUT_LINE_COIL_COMPLEX`` layout of radial lines.

An array in this layout has shape ``(slices, phases, readout, lines, coils, 2)``: ``L``
radial lines per slice and phase, each recorded by every coil, the last axis holding
the real and imaginary parts. Line ``l`` lies at ``l * pi / L``, and the center of
k-space at readout index ``readout // 2``. Nothing else about the array's origin
matters: measured and generated data, from any file format, read alike.

The MD-CNN radial cine dataset (H. El-Rewaidy, "Replication Data for: Multi-Domain
Convolutional Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac
MRI," Harvard Dataverse, 2020, doi:10.7910/DVN/CI3WB6, CC0 1.0) is stored this way,
as the variable ``data`` of MATLAB 5 files such as
``P01_slc1_phs25_lins196_cols832_cha16.mat``: 196 (sometimes 200 or 204) lines
per cardiac phase and a twice-oversampled readout zero-padded to 832 samples. The
dataset does not state its line angles or its center; both are measured, and the
direction of its first line relative to the patient is not recorded, so its images may
be a rotated or mirrored view of the scanner's. ``DESIGN_NOTES.md``, "Acquisition: the
MD-CNN dataset's line angles are measured," has the measurement.
"""

import numpy as np

from ._base import BaseRadialReader, RadialLineLayoutError

#: The selectable axes, in the array's own order, with their axis indices.
_SELECTABLE_AXES = {"slice": 0, "phase": 1, "coil": 4}

#: The readout and line axes.
_READOUT_AXIS, _LINE_AXIS = 2, 3

#: The axis holding the real and imaginary parts, and its size.
_PART_AXIS, _N_PARTS = 5, 2

#: The rank of an array in this layout.
_DATA_NDIM = 6


def _check_shape(shape: tuple[int, ...]) -> None:
    """Check that an array has the layout's rank and real/imaginary axis.

    :param shape: The array's shape.
    :type shape: tuple[int, ...]
    :raises RadialLineLayoutError: If the shape is not the layout's.

    """
    if len(shape) != _DATA_NDIM or shape[_PART_AXIS] != _N_PARTS:
        raise RadialLineLayoutError(
            "the array must be (slices, phases, readout, lines, coils, 2), got shape "
            f"{shape}"
        )


class SlicePhaseReadoutLineCoilComplexReader(BaseRadialReader):
    """Read arrays in the ``SLICE_PHASE_READOUT_LINE_COIL_COMPLEX`` layout.

    The selectable axes are ``slice``, ``phase`` and ``coil``, in that order.
    """

    @classmethod
    def _axis_sizes(cls, shape: tuple[int, ...]) -> dict[str, int]:
        """Give the sizes of ``slice``, ``phase`` and ``coil``.

        :param shape: The array's shape.
        :type shape: tuple[int, ...]
        :returns: The sizes of ``slice``, ``phase`` and ``coil``.
        :rtype: dict[str, int]
        :raises RadialLineLayoutError: If the shape is not the layout's.

        """
        _check_shape(shape=shape)
        return {name: shape[axis] for name, axis in _SELECTABLE_AXES.items()}

    @classmethod
    def _select_lines(
        cls, values: np.ndarray, *, index: tuple[int | slice, ...]
    ) -> np.ndarray:
        """Select the lines and join their real and imaginary parts.

        :param values: The array.
        :type values: np.ndarray
        :param index: One entry per selectable axis, ``slice``, ``phase``, ``coil``.
        :type index: tuple[int | slice, ...]
        :returns: The complex lines, ``(readout, line, *kept)``.
        :rtype: np.ndarray
        :raises RadialLineLayoutError: If the array is complex: its last axis holds
            the real and imaginary parts.

        """
        if np.iscomplexobj(values):
            raise RadialLineLayoutError(
                "the array must be real: its last axis holds the real and imaginary "
                "parts"
            )
        # The readout and line axes first, then slice, phase, coil, then the parts.
        lines = np.moveaxis(
            a=values,
            source=(_READOUT_AXIS, _LINE_AXIS) + tuple(_SELECTABLE_AXES.values()),
            destination=tuple(range(_PART_AXIS)),
        )
        # The selection, before the two parts are joined into one complex array.
        selected = lines[(slice(None), slice(None)) + index]
        return selected[..., 0] + 1j * selected[..., 1]

    @staticmethod
    def _line_angles(n_lines: int) -> np.ndarray:
        """Give line ``l`` its angle, ``l * pi / n_lines``.

        :param n_lines: The number of lines.
        :type n_lines: int
        :returns: The angles, in radians.
        :rtype: np.ndarray

        """
        return np.arange(n_lines) * (np.pi / n_lines)

    @staticmethod
    def _center_index(n_readout: int) -> int:
        """Give the center of k-space, readout index ``n_readout // 2``.

        :param n_readout: The number of readout samples per line.
        :type n_readout: int
        :returns: The center index.
        :rtype: int

        """
        return n_readout // 2
