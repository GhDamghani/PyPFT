# cython: language_level=3
"""
RULES:
    - Validators only validate the input arguments. They never mutate or replace them.
    - Validators are stateless methods.
    - Validators are designed to be composable. They can be combined to create more
        complex validation rules. For example, a validator for a list of integers can be
        composed of a type-validator for lists and a type-validator for integers.
    - Validators assume that the values have been type-validated, unless the validator
        is a type-validator itself. This is to avoid redundant type-checking and improve
        performance.
    - Validators raise ``TypeError`` for type violations, ``ValueError`` for value
        violations, and OS-level subclasses of ``OSError`` (e.g. ``PermissionError``,
        ``FileNotFoundError``) for filesystem state violations where a more specific OS
        exception is more appropriate than a generic ``ValueError``.
    - To avoid circular imports, validators for local library types are defined in the
      same class as they are defined.

CONVENTIONS:
    - Each validator is a standalone class named ``<Type>Validator`` (e.g.
        ``IntValidator``, ``PathValidator``). Import the specific class you need.
    - The order of classes and methods across the validator classes is inspired by the order of
        imports according to PEP8:
        - Simple types: built-in, standard library, ant then third-party libraries. Local
            types should be defined in the same class as the type definitions. The order
            of built-in types follows the standard type hierarchy laid out in the Python
            documentation:
                https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy

        - Composite types: If a composite type is validated, the composite type
            is listed in its collection types, after the simple-type validators. For
            example, ``tuple[int, ...]`` is listed in ``tuple`` because it is the type of
            the collection.

        - Validators with multiple input arguments: They should be treated similarly
            to inherited types and listed under the lowest type in the hierarchy. For
            example, ``value1_should_contain_value2(value1: list, value2: int)`` is listed
            under ``list`` because it is the lowest type of composition in the type
            hierarchy.
    - The order of validators in a class, as briefly discussed in the previous point, is:
      - Main outline:
        - Type validators
        - Value validators
        - Other validators
      - Within each validator category:
        - Simple-type
        - Composite-type
      - Within each type category:
        - Single-input
        - Multiple-input
    - Validators follow these name patterns:
      - Type validators: ``type_<'is'>_<typename>(value: <type>)``
      - Value validators: ``value_<'is' | 'has' | 'should' | ...>_<condition>(value: <type>)``
      - If composite type: use ``<simple_type>_<subtype>`` for <typename> and update <type>
        to the composite type. For example, ``type_is_tuple_int(value: tuple[int, ...])``.
      - If multiple input arguments: use ``value1``, ``value2``, etc. for the input arguments
        and update <condition> to reflect the relationship between the values. For example,
        ``value1_is_greater_than_value2(value1: int, value2: int)``.

DEVELOPMENT:
    VS Code snippets in ``.vscode/helpers.code-snippets``:
        - ``v-type`` — scaffolds a type-validator ``@classmethod`` with the standard
            RST docstring, ``isinstance`` validator, and ``TypeError`` raise.
        - ``v-value`` — scaffolds a value-validator ``@classmethod`` with the
            standard RST docstring and ``ValueError`` raise.
"""  # noqa: RST

import os
from enum import Enum, EnumType
from pathlib import Path
from typing import Any

import numpy as np

# ========================================================================================
# numbers.Integral: int, bool
# ========================================================================================


class IntValidator:
    """Validate integer (``int``) values."""

    @staticmethod
    def type_is_int(value: int) -> None:
        """Type-validator for int.

        :param value: The value to be validated.
        :type value: int
        :raises TypeError: If the value is not an int.

        """
        if not isinstance(value, int):
            raise TypeError(f"value must be {int.__name__}, got {type(value).__name__}")

    @staticmethod
    def value_is_non_negative(value: int) -> None:
        """Value-validator to check if an int is non-negative.

        :param value: The value to be validated.
        :type value: int
        :raises ValueError: If the value is negative.

        """
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}")


# ========================================================================================
# numbers.Real: float
# ========================================================================================


class FloatValidator:
    """Validate floating-point (``float``) values."""

    @staticmethod
    def type_is_float(value: float) -> None:
        """Type-validator for float.

        :param value: The value to be validated.
        :type value: float
        :raises TypeError: If the value is not a float.

        """
        if not isinstance(value, float):
            raise TypeError(
                f"value must be {float.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def value_is_positive(value: float) -> None:
        """Value-validator to check if a float is positive.

        :param value: The value to be validated.
        :type value: float
        :raises ValueError: If the value is not strictly positive.

        """
        if value <= 0:
            raise ValueError(f"value must be positive, got {value}")


# ========================================================================================
# Standard Library types
# ========================================================================================


class PathValidator:
    """Validate ``pathlib.Path`` values."""

    @staticmethod
    def type_is_Path(value: Path) -> None:
        """Type-validator for pathlib.Path.

        :param value: The value to be validated.
        :type value: Path
        :raises TypeError: If the value is not a pathlib.Path.

        """
        if not isinstance(value, Path):
            raise TypeError(
                f"value must be {Path.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def value_is_writable(value: Path) -> None:
        """Value-validator to check if a pathlib.Path is writable.

        If the path exists, it must be directly writable. If it does not
        exist, the validator walks up the directory tree to find the first
        existing ancestor and checks that it is a directory that is writable
        (i.e. the full path could be created). On POSIX, creating a file
        within a directory also requires execute (search) permission on that
        directory, so both ``W_OK`` and ``X_OK`` are checked for ancestors.

        :param value: The value to be validated.
        :type value: Path
        :raises FileNotFoundError: If no existing ancestor directory can be
            found on the filesystem (e.g. non-existent drive on Windows).
        :raises NotADirectoryError: If the closest existing ancestor is not
            a directory (e.g. a file), so no children can be created under it.
        :raises PermissionError: If the existing path or its closest existing
            ancestor is not writable.

        """
        if value.exists():
            if not os.access(value, os.W_OK):
                raise PermissionError(f"value must be a writable path, got {value}")
        else:
            ancestor = value.parent
            while not ancestor.exists():
                if ancestor == ancestor.parent:
                    raise FileNotFoundError(
                        f"value no existing ancestor directory found for: {value}"
                    )
                ancestor = ancestor.parent
            if not ancestor.is_dir():
                raise NotADirectoryError(
                    f"value ancestor is not a directory: {ancestor}"
                )
            if not os.access(ancestor, os.W_OK | os.X_OK):
                raise PermissionError(
                    f"value ancestor directory is not writable: {ancestor}"
                )

    @staticmethod
    def value_is_readable(value: Path) -> None:
        """Value-validator to check if a pathlib.Path is readable.

        :param value: The value to be validated.
        :type value: Path
        :raises FileNotFoundError: If the path does not exist.
        :raises PermissionError: If the path exists but is not readable.

        """
        if not value.exists():
            raise FileNotFoundError(f"value must exist, got {value}")
        if not os.access(value, os.R_OK):
            raise PermissionError(f"value must be a readable path, got {value}")

    @staticmethod
    def value_is_a_file(value: Path) -> None:
        """Value-validator to check if a pathlib.Path is a file.

        :param value: The value to be validated.
        :type value: Path
        :raises FileNotFoundError: If the path does not exist.
        :raises IsADirectoryError: If the path exists but is a directory.

        """
        if not value.exists():
            raise FileNotFoundError(f"value must exist, got {value}")
        if not value.is_file():
            raise IsADirectoryError(f"value must be a file, got {value}")

    @staticmethod
    def value_has_correct_suffix(value: Path, suffix: str) -> None:
        """Value-validator to check if a pathlib.Path has a specific suffix.

        :param value: The value to be validated.
        :type value: Path
        :param suffix: The expected suffix of the pathlib.Path.
        :type suffix: str
        :raises ValueError: If the value does not have the specified suffix.

        """
        # Compare case-insensitively so uppercase suffixes are also accepted.
        if value.suffix.lower() != suffix.lower():
            raise ValueError(f"value must have suffix {suffix}, got {value.suffix}")


class EnumValidator:
    """Validate enum values."""

    @staticmethod
    def type_is_enum(value: Enum) -> None:
        """Type-validator for enum.Enum.

        :param value: The value to be validated.
        :type value: Enum
        :raises TypeError: If the value is not an enum.Enum.

        """
        if not isinstance(value, Enum):
            raise TypeError(
                f"value must be {Enum.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def value_is_enum_member(value: Any, enum_class: EnumType) -> None:
        """Value-validator to check if a value is a member of a specific enum class.

        :param value: The value to be validated.
        :type value: Any
        :param enum_class: The expected enum class of the value.
        :type enum_class: EnumType
        :raises ValueError: If the value is not a member of the specified enum class.

        """
        try:
            enum_class(value)
        except ValueError:
            raise ValueError(
                f"value must be a member of {enum_class.__name__}, got {value}"
            )


# ========================================================================================
# Third-party libraries
# ========================================================================================


class NumpyValidator:
    """Validate ``numpy.ndarray`` values."""

    @staticmethod
    def type_is_ndarray(value: np.ndarray) -> None:
        """Type-validator for numpy.ndarray.

        :param value: The value to be validated.
        :type value: np.ndarray
        :raises TypeError: If the value is not a numpy.ndarray.

        """
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"value must be {np.ndarray.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def value_is_1d(value: np.ndarray) -> None:
        """Value-validator to check if a numpy.ndarray is one-dimensional.

        :param value: The value to be validated.
        :type value: np.ndarray
        :raises ValueError: If the value does not have exactly one dimension.

        """
        if value.ndim != 1:
            raise ValueError(f"value must be 1-D, got {value.ndim}-D")

    @staticmethod
    def value_is_at_least_1d(value: np.ndarray) -> None:
        """Value-validator to check if a numpy.ndarray has at least one dimension.

        Unlike ``value_is_1d``, this accepts arrays of any rank ``>= 1`` -- used by
        entry points that operate along a single named axis of an otherwise
        arbitrary-rank array (e.g. ``pypft.dht.hankel_transform``'s ``axis``
        argument), rather than requiring the whole array to be a bare vector.

        :param value: The value to be validated.
        :type value: np.ndarray
        :raises ValueError: If the value is 0-D (a scalar array).

        """
        if value.ndim < 1:
            raise ValueError(f"value must have at least 1 dimension, got {value.ndim}-D")

    @staticmethod
    def value_has_axis(value: np.ndarray, axis: int) -> None:
        """Value-validator to check if ``axis`` is a valid axis index for ``value``.

        :param value: The array ``axis`` is meant to index into.
        :type value: np.ndarray
        :param axis: The axis index to validate, ``numpy``-style (negative
            indices count from the end).
        :type axis: int
        :raises ValueError: If ``axis`` is out of bounds for ``value.ndim``.

        """
        if not (-value.ndim <= axis < value.ndim):
            raise ValueError(
                f"axis {axis} is out of bounds for {value.ndim}-D value"
            )

    @staticmethod
    def value1_axis_length_matches_value2(
        value1: np.ndarray, axis1: int, value2: np.ndarray, axis2: int
    ) -> None:
        """Value-validator to check two arrays agree in length along given axes.

        :param value1: The first array.
        :type value1: np.ndarray
        :param axis1: The axis of ``value1`` to compare.
        :type axis1: int
        :param value2: The second array.
        :type value2: np.ndarray
        :param axis2: The axis of ``value2`` to compare.
        :type axis2: int
        :raises ValueError: If ``value1.shape[axis1] != value2.shape[axis2]``.

        """
        length1 = value1.shape[axis1]
        length2 = value2.shape[axis2]
        if length1 != length2:
            raise ValueError(
                f"value1 has length {length1} along axis {axis1}, but value2 has "
                f"length {length2} along axis {axis2}"
            )
