"""Arrays stored in files, read whatever the layout of their axes.

A file holds either a single array or a collection of named arrays:

- ``.npy``: a single array, read as it is.
- ``.npz``: named arrays, one of which is chosen by name.
- ``.mat`` (MATLAB 5, read by ``scipy.io``): named variables, one of which is chosen
  by name. MATLAB 7.3 files are HDF5 files, which ``scipy.io`` does not read.

``stored_arrays`` lists a collection's numeric arrays and their shapes from the file's
headers, without loading their data, so a caller can offer the choice;
``load_stored_array`` reads one. Nothing here knows about radial lines: the axes of
the array that comes back are interpreted by a ``pypft.acquisition.RadialLineLayout``.
"""

import zipfile
from pathlib import Path
from typing import IO

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

from pypft.utils.validators import PathValidator, StrValidator

#: The suffix of a file holding a single array.
NPY_SUFFIX = ".npy"

#: The suffix of a NumPy file holding named arrays.
NPZ_SUFFIX = ".npz"

#: The suffix of a MATLAB file holding named variables.
MAT_SUFFIX = ".mat"

#: The suffixes of the files holding several named arrays.
MULTI_ARRAY_SUFFIXES = (NPZ_SUFFIX, MAT_SUFFIX)

#: Every suffix ``load_stored_array`` reads.
ARRAY_FILE_SUFFIXES = (NPY_SUFFIX, *MULTI_ARRAY_SUFFIXES)

#: The ``numpy.dtype.kind`` codes of numeric arrays: booleans, integers, floats and
#: complex numbers.
_NUMERIC_KINDS = "biufc"

#: The MATLAB classes ``scipy.io.whosmat`` reports for numeric variables.
_MATLAB_NUMERIC_CLASSES = frozenset(
    {
        "double",
        "single",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "logical",
    }
)

#: What reading a damaged, empty or unsupported file raises: ``scipy.io`` on a file
#: that is no MATLAB 5 file (``IndexError`` below its 128-byte header,
#: ``NotImplementedError`` for MATLAB 7.3), and ``numpy`` or ``zipfile`` on a broken
#: ``.npy`` or ``.npz`` file.
_UNREADABLE_ERRORS = (
    MatReadError,
    NotImplementedError,
    ValueError,
    IndexError,
    EOFError,
    zipfile.BadZipFile,
)


def _validate_file(path: Path) -> str:
    """Validate a path to a readable array file and return its lowercase suffix.

    :param path: The file.
    :type path: Path
    :returns: The suffix.
    :rtype: str
    :raises TypeError: If ``path`` is not a ``pathlib.Path``.
    :raises ValueError: If the suffix is not one of ``ARRAY_FILE_SUFFIXES``.
    :raises OSError: If ``path`` does not exist, is a directory, or cannot be read.

    """
    PathValidator.type_is_Path(value=path)
    PathValidator.value_has_suffix_in(value=path, suffixes=ARRAY_FILE_SUFFIXES)
    PathValidator.value_is_a_file(value=path)
    PathValidator.value_is_readable(value=path)
    return path.suffix.lower()


def holds_several_arrays(path: Path) -> bool:
    """Tell whether a file holds named arrays to choose from, by its suffix.

    :param path: The file.
    :type path: Path
    :returns: ``True`` for a ``.npz`` or ``.mat`` file, ``False`` for a ``.npy`` file.
    :rtype: bool
    :raises TypeError: If ``path`` is not a ``pathlib.Path``.
    :raises ValueError: If the suffix is not one of ``ARRAY_FILE_SUFFIXES``.

    """
    PathValidator.type_is_Path(value=path)
    PathValidator.value_has_suffix_in(value=path, suffixes=ARRAY_FILE_SUFFIXES)
    return path.suffix.lower() in MULTI_ARRAY_SUFFIXES


def _npy_header(file: IO[bytes]) -> tuple[tuple[int, ...], np.dtype]:
    """Read the shape and type of a ``.npy`` array from its header.

    :param file: The open array, at its start.
    :type file: IO[bytes]
    :returns: The shape and the element type.
    :rtype: tuple[tuple[int, ...], np.dtype]

    """
    version = np.lib.format.read_magic(file)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(file)
    else:
        shape, _, dtype = np.lib.format.read_array_header_2_0(file)
    return tuple(shape), dtype


def _npz_arrays(path: Path) -> dict[str, tuple[int, ...]]:
    """List a ``.npz`` file's numeric arrays from their headers."""
    arrays: dict[str, tuple[int, ...]] = {}
    with zipfile.ZipFile(file=path) as archive:
        for member in archive.namelist():
            if not member.endswith(NPY_SUFFIX):
                continue
            with archive.open(name=member) as file:
                shape, dtype = _npy_header(file=file)
            if dtype.kind in _NUMERIC_KINDS:
                arrays[member.removesuffix(NPY_SUFFIX)] = shape
    return arrays


def _mat_arrays(path: Path) -> dict[str, tuple[int, ...]]:
    """List a MATLAB 5 file's numeric variables from their headers."""
    return {
        name: tuple(shape)
        for name, shape, matlab_class in whosmat(file_name=path)
        if matlab_class in _MATLAB_NUMERIC_CLASSES
    }


def stored_arrays(path: Path) -> dict[str, tuple[int, ...]]:
    """List the numeric arrays of a file holding several, with their shapes.

    Only the headers are read, never the arrays' data. Non-numeric entries (MATLAB
    structs, cells and text, object arrays) are left out, since no layout reads them.

    :param path: A ``.npz`` or ``.mat`` file.
    :type path: Path
    :returns: Every numeric array's name and shape, in the file's own order.
    :rtype: dict[str, tuple[int, ...]]
    :raises TypeError: If ``path`` is not a ``pathlib.Path``.
    :raises ValueError: If the file holds a single array (``.npy``), has an
        unsupported suffix, cannot be read as its format, or holds no numeric array.
    :raises OSError: If ``path`` does not exist, is a directory, or cannot be read.

    """
    suffix = _validate_file(path=path)
    if suffix not in MULTI_ARRAY_SUFFIXES:
        raise ValueError(f"{path.name} holds a single array: read it without a name")
    try:
        arrays = (
            _npz_arrays(path=path) if suffix == NPZ_SUFFIX else _mat_arrays(path=path)
        )
    except _UNREADABLE_ERRORS as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error
    if not arrays:
        raise ValueError(f"{path.name} holds no numeric array")
    return arrays


def _load_npz_array(path: Path, name: str) -> np.ndarray:
    """Read one named array of a ``.npz`` file."""
    # np.load leaves a broken archive's file open, so the archive is checked first.
    if not zipfile.is_zipfile(filename=path):
        raise ValueError(f"not a {NPZ_SUFFIX} archive")
    # allow_pickle=False: an array is read as plain data, never as code.
    with np.load(file=path, allow_pickle=False) as archive:
        if name not in archive.files:
            raise KeyError(name)
        return archive[name]


def _load_mat_array(path: Path, name: str) -> np.ndarray:
    """Read one named variable of a MATLAB 5 file."""
    variables = loadmat(file_name=path, variable_names=[name])
    if name not in variables:
        raise KeyError(name)
    return variables[name]


def load_stored_array(path: Path, *, name: str | None = None) -> np.ndarray:
    """Read one numeric array from a file.

    A ``.npy`` file's single array is read with ``name`` omitted; a ``.npz`` or
    ``.mat`` file's array is chosen by ``name``, one of ``stored_arrays``' keys.

    :param path: The file.
    :type path: Path
    :param name: The array to read from a file holding several; ``None`` for a file
        holding one.
    :type name: str | None
    :returns: The array, as stored.
    :rtype: np.ndarray
    :raises TypeError: If ``path`` is not a ``pathlib.Path`` or ``name`` is neither a
        ``str`` nor ``None``.
    :raises ValueError: If the suffix is unsupported, ``name`` is missing for a file
        holding several arrays or given for one holding one, the file holds no array
        of that name, cannot be read as its format, or the array is not numeric.
    :raises OSError: If ``path`` does not exist, is a directory, or cannot be read.

    """
    suffix = _validate_file(path=path)
    several = suffix in MULTI_ARRAY_SUFFIXES
    if name is not None:
        StrValidator.type_is_str(value=name)
    if several and name is None:
        raise ValueError(f"{path.name} holds several arrays: choose one by name")
    if not several and name is not None:
        raise ValueError(f"{path.name} holds a single array: read it without a name")
    try:
        if name is None:
            # The file is opened here so it is closed whatever np.load raises.
            with path.open(mode="rb") as file:
                values = np.load(file=file, allow_pickle=False)
        elif suffix == NPZ_SUFFIX:
            values = _load_npz_array(path=path, name=name)
        else:
            values = _load_mat_array(path=path, name=name)
    except KeyError:
        raise ValueError(f"{path.name} holds no array named {name!r}") from None
    except _UNREADABLE_ERRORS as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error
    if not isinstance(values, np.ndarray) or values.dtype.kind not in _NUMERIC_KINDS:
        stored = path.name if name is None else f"{path.name}'s array {name!r}"
        raise ValueError(f"{stored} is not a numeric array")
    return values
