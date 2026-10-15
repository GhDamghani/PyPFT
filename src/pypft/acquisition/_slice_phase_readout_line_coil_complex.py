"""Reader for the ``SLICE_PHASE_READOUT_LINE_COIL_COMPLEX`` layout of radial lines.

A file in this layout is a MATLAB 5 ``.mat`` file holding one variable, ``data``, of
shape ``(slices, phases, readout, lines, coils, 2)``: ``L`` radial lines per slice and
phase, each recorded by every coil, the last axis holding the real and imaginary
parts. Line ``l`` lies at ``l * pi / L``, and the center of k-space at readout index
``readout // 2``. Nothing else about the file's origin matters: measured and generated
data in this layout read alike.

The MD-CNN radial cine dataset (H. El-Rewaidy, "Replication Data for: Multi-Domain
Convolutional Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac
MRI," Harvard Dataverse, 2020, doi:10.7910/DVN/CI3WB6, CC0 1.0) is stored this way,
e.g. ``P01_slc1_phs25_lins196_cols832_cha16.mat``: 196 (sometimes 200 or 204) lines
per cardiac phase and a twice-oversampled readout zero-padded to 832 samples. The
dataset does not state its line angles or its center; both are measured, and the
direction of its first line relative to the patient is not recorded, so its images may
be a rotated or mirrored view of the scanner's. ``DESIGN_NOTES.md``, "Acquisition: the
MD-CNN dataset's line angles are measured," has the measurement.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

from ._base import BaseRadialReader, RadialLineLayoutError

#: The name of the layout's only variable.
_VARIABLE = "data"

#: The selectable axes of ``data``, in its own order, with their axis indices.
_SELECTABLE_AXES = {"slice": 0, "phase": 1, "coil": 4}

#: The readout and line axes of ``data``.
_READOUT_AXIS, _LINE_AXIS = 2, 3

#: The axis of ``data`` holding the real and imaginary parts, and its size.
_PART_AXIS, _N_PARTS = 5, 2

#: The rank of ``data``.
_DATA_NDIM = 6

#: What ``scipy.io`` raises for a file it cannot read as a MATLAB 5 file: a damaged
#: or empty file, one shorter than the 128-byte header (``IndexError``), or a MATLAB
#: 7.3 (HDF5) file.
_UNREADABLE_ERRORS = (MatReadError, NotImplementedError, ValueError, IndexError)


def _check_shape(shape: tuple[int, ...], path: Path) -> None:
    """Check that ``data`` has the layout's rank and real/imaginary axis.

    :param shape: The shape of the file's ``data`` variable.
    :type shape: tuple[int, ...]
    :param path: The file, for the error message.
    :type path: Path
    :raises RadialLineLayoutError: If the shape is not the layout's.

    """
    if len(shape) != _DATA_NDIM or shape[_PART_AXIS] != _N_PARTS:
        raise RadialLineLayoutError(
            f"{path}: {_VARIABLE!r} must be (slices, phases, readout, lines, coils, "
            f"2), got shape {shape}"
        )


class SlicePhaseReadoutLineCoilComplexReader(BaseRadialReader):
    """Read ``.mat`` files in the ``SLICE_PHASE_READOUT_LINE_COIL_COMPLEX`` layout.

    The selectable axes are ``slice``, ``phase`` and ``coil``, in that order.
    """

    SUFFIX = ".mat"

    @classmethod
    def _axis_sizes(cls, path: Path) -> dict[str, int]:
        """Read the selectable axes' sizes from the file's header, without its data.

        :param path: The file to inspect.
        :type path: Path
        :returns: The sizes of ``slice``, ``phase`` and ``coil``.
        :rtype: dict[str, int]
        :raises RadialLineLayoutError: If the file is not a readable MATLAB 5 file,
            or has no ``data`` variable of the layout's shape.

        """
        try:
            variables = whosmat(file_name=path)
        except _UNREADABLE_ERRORS as error:
            raise RadialLineLayoutError(
                f"{path}: not a readable MATLAB 5 file: {error}"
            ) from error
        shapes = {name: shape for name, shape, _ in variables}
        if _VARIABLE not in shapes:
            raise RadialLineLayoutError(
                f"{path} holds no {_VARIABLE!r} variable, only {sorted(shapes)}"
            )
        shape = tuple(shapes[_VARIABLE])
        _check_shape(shape=shape, path=path)
        return {name: shape[axis] for name, axis in _SELECTABLE_AXES.items()}

    @classmethod
    def _load_lines(cls, path: Path, *, index: tuple[int | slice, ...]) -> np.ndarray:
        """Load the selected lines and join their real and imaginary parts.

        :param path: The file to read.
        :type path: Path
        :param index: One entry per selectable axis, ``slice``, ``phase``, ``coil``.
        :type index: tuple[int | slice, ...]
        :returns: The complex lines, ``(readout, line, *kept)``.
        :rtype: np.ndarray
        :raises RadialLineLayoutError: If the file is not a readable MATLAB 5 file,
            or ``data`` is not of the layout's shape.

        """
        try:
            data = loadmat(file_name=path, variable_names=[_VARIABLE])[_VARIABLE]
        except _UNREADABLE_ERRORS as error:
            raise RadialLineLayoutError(
                f"{path}: not a readable MATLAB 5 file: {error}"
            ) from error
        _check_shape(shape=data.shape, path=path)
        # The readout and line axes first, then slice, phase, coil, then the parts.
        lines = np.moveaxis(
            a=data,
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
