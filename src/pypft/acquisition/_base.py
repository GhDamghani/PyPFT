"""The base class every radial-line-layout reader is built on.

A reader turns files of one array layout into a ``pypft.acquisition.RadialKSpace``.
Every layout stores its k-space lines with its own axes: the readout and the line, plus
any number of others (slices, cardiac phases, coils, ...). Those others are
*selectable*: a selection picks one index of each (a single sample) or keeps the whole
axis (a batch).
``BaseRadialReader`` owns that shared logic as template methods, ``selectable_axes``
and ``read``; a layout overrides four hooks only:

- ``_axis_sizes(path)``: the size of every selectable axis, read cheaply from the
  file's header where the layout allows it.
- ``_load_lines(path, index=...)``: the complex lines ``(readout, line, *kept)``, with
  the selection applied while loading, so a large file is never held in memory whole
  as complex numbers.
- ``_line_angles(n_lines)``: the angle of every line, in PyPFT's convention.
- ``_center_index(n_readout)``: the readout index of the center of k-space.

Like ``pypft.dht``'s strategies, readers are stateless: every method is a classmethod
or staticmethod, and a reader is never instantiated.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np

from pypft.utils.validators import (
    IntValidator,
    MappingValidator,
    PathValidator,
    StrValidator,
)

from ._kspace import RadialKSpace, radial_lines_to_polar

#: The index a selectable axis takes when a selection omits it.
_DEFAULT_INDEX = 0

#: The leading axes of every ``_load_lines`` result: the readout and the line.
_LINES_SAMPLE_NDIM = 2


class RadialLineLayoutError(ValueError):
    """Raised when a file does not hold its array in the layout chosen for it."""


class BaseRadialReader:
    """Template for reading files of one radial-line layout into a ``RadialKSpace``."""

    #: The file suffix of the layout, with its leading dot.
    SUFFIX: ClassVar[str]

    # ==================================================================================
    # Hooks
    # ==================================================================================

    @classmethod
    def _axis_sizes(cls, path: Path) -> dict[str, int]:
        """Read the size of every selectable axis, in the layout's own axis order.

        :param path: The file to inspect, already validated.
        :type path: Path
        :returns: Every selectable axis's name and size.
        :rtype: dict[str, int]
        :raises RadialLineLayoutError: If the file is not in this layout.
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @classmethod
    def _load_lines(cls, path: Path, *, index: tuple[int | slice, ...]) -> np.ndarray:
        """Load the selected complex k-space lines.

        :param path: The file to read, already validated.
        :type path: Path
        :param index: One entry per selectable axis, in ``_axis_sizes``' order: an
            index to take, or ``slice(None)`` to keep the axis.
        :type index: tuple[int | slice, ...]
        :returns: The lines, ``(readout, line, *kept)``, the kept axes in order.
        :rtype: np.ndarray
        :raises RadialLineLayoutError: If the file is not in this layout.
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @staticmethod
    def _line_angles(n_lines: int) -> np.ndarray:
        """Give the angle of every line's samples past its center.

        :param n_lines: The number of lines.
        :type n_lines: int
        :returns: The angles, in radians, in PyPFT's convention.
        :rtype: np.ndarray
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @staticmethod
    def _center_index(n_readout: int) -> int:
        """Give the readout index of the center of k-space.

        :param n_readout: The number of readout samples per line.
        :type n_readout: int
        :returns: The center index.
        :rtype: int
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    # ==================================================================================
    # Template methods
    # ==================================================================================

    @classmethod
    def _validate_path(cls, path: Path) -> None:
        """Validate a file path for this layout.

        :param path: The path to validate.
        :type path: Path
        :raises TypeError: If ``path`` is not a ``pathlib.Path``.
        :raises ValueError: If ``path`` does not have the layout's suffix.
        :raises OSError: If ``path`` does not exist, is a directory, or cannot be
            read.

        """
        PathValidator.type_is_Path(value=path)
        PathValidator.value_has_correct_suffix(value=path, suffix=cls.SUFFIX)
        PathValidator.value_is_a_file(value=path)
        PathValidator.value_is_readable(value=path)

    @classmethod
    def selectable_axes(cls, path: Path) -> dict[str, int]:
        """Read the name and size of every selectable axis of a file.

        :param path: The file to inspect.
        :type path: Path
        :returns: Every selectable axis's name and size, in the layout's axis order.
        :rtype: dict[str, int]
        :raises TypeError: If ``path`` is not a ``pathlib.Path``.
        :raises ValueError: If ``path`` has the wrong suffix.
        :raises OSError: If ``path`` cannot be read.
        :raises RadialLineLayoutError: If the file is not in this layout.

        """
        cls._validate_path(path=path)
        return cls._axis_sizes(path=path)

    @staticmethod
    def _selection_index(
        sizes: dict[str, int], selection: Mapping[str, int | None]
    ) -> tuple[int | slice, ...]:
        """Turn a selection into one index entry per selectable axis.

        :param sizes: Every selectable axis's name and size.
        :type sizes: dict[str, int]
        :param selection: The selection to apply.
        :type selection: Mapping[str, int | None]
        :returns: One entry per axis, in ``sizes``' order.
        :rtype: tuple[int | slice, ...]
        :raises TypeError: If a key is not a ``str`` or a value is neither an ``int``
            nor ``None``.
        :raises ValueError: If a key names no selectable axis or an index is outside
            its axis.

        """
        MappingValidator.type_is_mapping(value=selection)
        for name, choice in selection.items():
            StrValidator.type_is_str(value=name)
            if name not in sizes:
                raise ValueError(
                    f"selection names an unknown axis {name!r}; the selectable axes "
                    f"are {list(sizes)}"
                )
            if choice is not None:
                IntValidator.type_is_int(value=choice)
                IntValidator.value_is_non_negative(value=choice)
                IntValidator.value1_is_less_than_value2(
                    value1=choice, value2=sizes[name]
                )
        index: list[int | slice] = []
        for name in sizes:
            choice = selection.get(name, _DEFAULT_INDEX)
            # None keeps the whole axis; an int takes one index and drops the axis.
            index.append(slice(None) if choice is None else choice)
        return tuple(index)

    @classmethod
    def read(
        cls, path: Path, *, selection: Mapping[str, int | None] | None = None
    ) -> RadialKSpace:
        """Read the selected k-space of a file into PyPFT's uniform polar layout.

        ``selection`` maps a selectable axis's name to one index of it, or to
        ``None`` to keep the whole axis; an axis it omits takes index ``0``. With
        every axis given an index, the result is a single sample ``(radial,
        angular)``. Every kept axis goes into one batch axis, ``(radial, angular,
        batch)``, flattened in C order over the kept axes in the layout's own order
        (the last kept axis varying fastest), so ``values.reshape(radial, angular,
        *sizes)`` recovers them.

        :param path: The file to read.
        :type path: Path
        :param selection: The index or ``None`` per selectable axis; every axis takes
            index ``0`` when omitted.
        :type selection: Mapping[str, int | None] | None
        :returns: The selected k-space, converted by
            ``pypft.acquisition.radial_lines_to_polar``.
        :rtype: RadialKSpace
        :raises TypeError: If any argument has the wrong type.
        :raises ValueError: If ``path`` has the wrong suffix, or the selection names an
            unknown axis or an index outside its axis.
        :raises OSError: If ``path`` cannot be read.
        :raises RadialLineLayoutError: If the file is not in this layout.

        """
        sizes = cls.selectable_axes(path=path)
        index = cls._selection_index(
            sizes=sizes, selection={} if selection is None else selection
        )
        lines = cls._load_lines(path=path, index=index)
        n_readout, n_lines = lines.shape[:_LINES_SAMPLE_NDIM]
        if lines.ndim > _LINES_SAMPLE_NDIM:
            # Every kept axis, flattened into PyPFT's single trailing batch axis.
            lines = lines.reshape((n_readout, n_lines, -1))
        return radial_lines_to_polar(
            lines=lines,
            angles=cls._line_angles(n_lines=n_lines),
            center_index=cls._center_index(n_readout=n_readout),
        )
