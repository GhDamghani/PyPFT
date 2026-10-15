r"""Cut a small test fixture out of one file of the MD-CNN radial cine dataset.

The MD-CNN dataset (H. El-Rewaidy, "Replication Data for: Multi-Domain Convolutional
Neural Network (MD-CNN) For Radial Reconstruction of Dynamic Cardiac MRI," Harvard
Dataverse, 2020, doi:10.7910/DVN/CI3WB6, released under CC0 1.0) holds radial cine
k-space of 108 subjects, one MATLAB ``.mat`` file each of about 230 MB. A file holds a
single variable, ``data``, of shape ``(slices, phases, readout, lines, coils, 2)``: the
readout is twice oversampled and zero-padded to 832 samples, each line is one radial
view through the center of k-space, and the last axis holds the real and imaginary
parts.

The fixture keeps that exact layout, so ``pypft.acquisition`` reads it with the same
reader as a full file, but only:

- the first slice and the first cardiac phase;
- the central ``--readout`` samples of every line (the zero-padding outside them is
  dropped, which keeps the k-space sample spacing and the center index
  ``readout // 2``);
- the ``--coils`` coils with the most signal energy, in ascending coil order;
- single-precision floats, written with compression.

Usage:
    uv run python scripts/make_md_cnn_excerpt.py [--source <P01 .mat file>] \
        [--readout 416] [--coils 4] [--output <fixture path>]

The committed fixture is exactly the output of the defaults, from the dataset's
``P01_slc1_phs25_lins196_cols832_cha16.mat``, which is not part of this repository:

    uv run python scripts/make_md_cnn_excerpt.py
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from pypft.utils.validators import IntValidator, NumpyValidator, PathValidator

# ========================================================================================
# Constants
# ========================================================================================

#: The repository root, two levels above this script.
_REPOSITORY = Path(__file__).resolve().parents[1]

#: Where the full dataset file is expected by default: a gitignored local folder.
DEFAULT_SOURCE = (
    _REPOSITORY
    / ".local_files"
    / "samples"
    / "El-Rewaidy, Hossam, 2020"
    / "P01_slc1_phs25_lins196_cols832_cha16.mat"
)

#: Readout samples kept around the center of every line.
DEFAULT_READOUT = 416

#: Coils kept, the ones with the most signal energy.
DEFAULT_COILS = 4

#: Where the committed fixture lives.
DEFAULT_OUTPUT = (
    _REPOSITORY / "tests" / "samples" / "P01_slc1_phs1_lins196_cols416_cha4.mat"
)

#: The name of the dataset's only variable.
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
    """Cut the fixture out of a full ``data`` array, keeping its layout.

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


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, cut the excerpt and write it.

    :param argv: The command-line arguments; ``sys.argv[1:]`` when omitted.
    :type argv: list[str] | None

    """
    parser = argparse.ArgumentParser(
        description="Cut a small test fixture out of an MD-CNN radial cine file."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="The full dataset file (default: the P01 file under .local_files).",
    )
    parser.add_argument(
        "--readout",
        type=int,
        default=DEFAULT_READOUT,
        help=f"Readout samples kept around the center (default: {DEFAULT_READOUT}).",
    )
    parser.add_argument(
        "--coils",
        type=int,
        default=DEFAULT_COILS,
        help=f"Coils kept, the strongest ones (default: {DEFAULT_COILS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where the fixture is written (default: tests/samples/).",
    )
    arguments = parser.parse_args(args=argv)

    PathValidator.value_is_a_file(value=arguments.source)
    PathValidator.value_is_readable(value=arguments.source)
    PathValidator.value_has_correct_suffix(value=arguments.output, suffix=".mat")
    PathValidator.value_is_writable(value=arguments.output)

    data = loadmat(file_name=arguments.source, variable_names=[VARIABLE])[VARIABLE]
    excerpt = make_excerpt(data=data, readout=arguments.readout, coils=arguments.coils)
    savemat(file_name=arguments.output, mdict={VARIABLE: excerpt}, do_compression=True)
    print(f"Wrote {arguments.output} {excerpt.shape}")


if __name__ == "__main__":
    main()
