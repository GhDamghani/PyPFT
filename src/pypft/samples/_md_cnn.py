r"""The MD-CNN radial cine excerpt: one small sample cut from one dataset file.

The MD-CNN dataset (H. El-Rewaidy, "Replication Data for: Multi-Domain Convolutional
Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac MRI," Harvard
Dataverse, 2020, doi:10.7910/DVN/CI3WB6, released under CC0 1.0) holds radial cine
k-space of 108 subjects, one MATLAB ``.mat`` file each of about 230 MB. A file holds a
single variable, ``data``, of shape ``(slices, phases, readout, lines, coils, 2)``: the
readout is twice oversampled and zero-padded to 832 samples, each line is one radial
view through the center of k-space, and the last axis holds the real and imaginary
parts.

The excerpt keeps that exact layout, so ``pypft.acquisition`` reads it with the same
reader as a full file, but only:

- the first slice and the first cardiac phase;
- the central ``EXCERPT_READOUT`` samples of every line (the zero-padding outside them
  is dropped, which keeps the k-space sample spacing and the center index
  ``readout // 2``);
- the ``EXCERPT_COILS`` coils with the most signal energy, in ascending coil order;
- single-precision floats, written with compression.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from pypft.utils.validators import IntValidator, NumpyValidator

# ========================================================================================
# Constants
# ========================================================================================

#: The dataset file the excerpt is cut from, subject P01, on Harvard Dataverse.
SOURCE_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId"
    "?persistentId=doi:10.7910/DVN/CI3WB6/XS8ZVS"
)

#: That file's name and its MD5 checksum, as Dataverse publishes it.
SOURCE_FILENAME = "P01_slc1_phs25_lins196_cols832_cha16.mat"
SOURCE_MD5 = "e1a7ff0a42e66e998bae71413c12e698"

#: Readout samples kept around the center of every line.
EXCERPT_READOUT = 416

#: Coils kept, the ones with the most signal energy.
EXCERPT_COILS = 4

#: The name of the dataset's only variable, kept by the excerpt.
VARIABLE = "data"

#: The rank of the dataset's ``data`` variable, and its axis indices.
_DATA_NDIM = 6
_SLICE_AXIS, _PHASE_AXIS, _READOUT_AXIS, _LINE_AXIS, _COIL_AXIS, _PART_AXIS = range(
    _DATA_NDIM
)

# ========================================================================================
# The excerpt
# ========================================================================================


def make_excerpt(data: np.ndarray, *, readout: int, coils: int) -> np.ndarray:
    """Cut the excerpt out of a full ``data`` array, keeping its layout.

    :param data: The dataset's ``(slices, phases, readout, lines, coils, 2)`` array.
    :type data: np.ndarray
    :param readout: The readout samples to keep, centered on index
        ``data.shape[2] // 2``; must be even so the kept center stays at
        ``readout // 2``.
    :type readout: int
    :param coils: The number of coils to keep.
    :type coils: int
    :returns: The ``(1, 1, readout, lines, coils, 2)`` single-precision excerpt.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``data`` is not of the dataset's rank, ``readout`` is not
        positive, odd or longer than the data's readout, or ``coils`` is not positive
        or exceeds the data's coil count.

    """
    NumpyValidator.type_is_ndarray(value=data)
    NumpyValidator.value_has_ndim_in(value=data, ndims=(_DATA_NDIM,))
    IntValidator.type_is_int(value=readout)
    IntValidator.value_is_positive(value=readout)
    IntValidator.type_is_int(value=coils)
    IntValidator.value_is_positive(value=coils)
    if readout % 2 or readout > data.shape[_READOUT_AXIS]:
        raise ValueError(
            f"readout must be even and at most {data.shape[_READOUT_AXIS]}, "
            f"got {readout}"
        )
    if coils > data.shape[_COIL_AXIS]:
        raise ValueError(f"coils must be at most {data.shape[_COIL_AXIS]}, got {coils}")
    # The first slice and phase, keeping both axes.
    frame = data[:1, :1]
    # Center index c of the full readout; the kept window [c - readout/2, c +
    # readout/2) puts that center at readout // 2 of the excerpt.
    start = data.shape[_READOUT_AXIS] // 2 - readout // 2
    window = frame[:, :, slice(start, start + readout)]
    # Energy per coil, over every other axis; the strongest coils, in coil order.
    energy = np.sum(
        a=window**2,
        axis=(_SLICE_AXIS, _PHASE_AXIS, _READOUT_AXIS, _LINE_AXIS, _PART_AXIS),
    )
    strongest = np.sort(a=np.argsort(a=energy)[::-1][:coils])
    return window[:, :, :, :, strongest].astype(np.float32)


def write_excerpt(source: Path, output: Path) -> None:
    """Read a dataset file, cut the excerpt and write it as a compressed ``.mat``.

    :param source: The full dataset file.
    :type source: Path
    :param output: Where the excerpt is written.
    :type output: Path

    """
    data = loadmat(file_name=source, variable_names=[VARIABLE])[VARIABLE]
    excerpt = make_excerpt(data=data, readout=EXCERPT_READOUT, coils=EXCERPT_COILS)
    savemat(file_name=output, mdict={VARIABLE: excerpt}, do_compression=True)
