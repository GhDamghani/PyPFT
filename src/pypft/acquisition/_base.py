"""The base class every radial-line-layout reader is built on.

A reader turns an array of one layout into a ``pypft.acquisition.RadialKSpace``. Every
layout stores its k-space lines with its own axes: the readout and the line, plus any
number of others (slices, cardiac phases, coils, ...). Those others are *selectable*: a
selection picks one index of each (a single sample) or keeps the whole axis (a batch).
A layout describes the array only, never the file it came from: any array, read by
``pypft.acquisition.load_stored_array`` or built in memory, is read alike.
``BaseRadialReader`` owns the shared logic as template methods, ``selectable_axes``
and ``read``; a layout overrides four hooks only:

- ``_axis_sizes(shape)``: the size of every selectable axis, from the array's shape.
- ``_select_lines(values, index=...)``: the complex lines ``(readout, line, *kept)``,
  with the selection applied before the lines are made complex, so a large array is
  never copied whole.
- ``_line_angles(n_lines)``: the angle of every line, in PyPFT's convention.
- ``_center_index(n_readout)``: the readout index of the center of k-space.

Like ``pypft.dht``'s strategies, readers are stateless: every method is a classmethod
or staticmethod, and a reader is never instantiated.
"""

from collections.abc import Mapping

import numpy as np

from pypft.utils.validators import (
    IntValidator,
    MappingValidator,
    NumpyValidator,
    StrValidator,
)

from ._kspace import RadialKSpace, radial_lines_to_polar

#: The index a selectable axis takes when a selection omits it.
_DEFAULT_INDEX = 0

#: The leading axes of every ``_select_lines`` result: the readout and the line.
_LINES_SAMPLE_NDIM = 2


class RadialLineLayoutError(ValueError):
    """Raised when an array is not laid out as the layout chosen for it."""


class BaseRadialReader:
    """Template for reading arrays of one radial-line layout into a ``RadialKSpace``."""

    # ==================================================================================
    # Hooks
    # ==================================================================================

    @classmethod
    def _axis_sizes(cls, shape: tuple[int, ...]) -> dict[str, int]:
        """Give the size of every selectable axis, in the layout's own axis order.

        :param shape: The array's shape.
        :type shape: tuple[int, ...]
        :returns: Every selectable axis's name and size.
        :rtype: dict[str, int]
        :raises RadialLineLayoutError: If the shape is not this layout's.
        :raises NotImplementedError: Always; subclasses must override this hook.

        """
        raise NotImplementedError

    @classmethod
    def _select_lines(
        cls, values: np.ndarray, *, index: tuple[int | slice, ...]
    ) -> np.ndarray:
        """Select the complex k-space lines.

        :param values: The array, already checked by ``_axis_sizes``.
        :type values: np.ndarray
        :param index: One entry per selectable axis, in ``_axis_sizes``' order: an
            index to take, or ``slice(None)`` to keep the axis.
        :type index: tuple[int | slice, ...]
        :returns: The lines, ``(readout, line, *kept)``, the kept axes in order.
        :rtype: np.ndarray
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
    def selectable_axes(cls, values: np.ndarray) -> dict[str, int]:
        """Give the name and size of every selectable axis of an array.

        :param values: The array.
        :type values: np.ndarray
        :returns: Every selectable axis's name and size, in the layout's axis order.
        :rtype: dict[str, int]
        :raises TypeError: If ``values`` is not a ``numpy.ndarray``.
        :raises RadialLineLayoutError: If the array is not in this layout.

        """
        NumpyValidator.type_is_ndarray(value=values)
        return cls._axis_sizes(shape=values.shape)

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
        cls, values: np.ndarray, *, selection: Mapping[str, int | None] | None = None
    ) -> RadialKSpace:
        """Read the selected k-space of an array into PyPFT's uniform polar layout.

        ``selection`` maps a selectable axis's name to one index of it, or to
        ``None`` to keep the whole axis; an axis it omits takes index ``0``. With
        every axis given an index, the result is a single sample ``(radial,
        angular)``. Every kept axis goes into one batch axis, ``(radial, angular,
        batch)``, flattened in C order over the kept axes in the layout's own order
        (the last kept axis varying fastest), so ``values.reshape(radial, angular,
        *sizes)`` recovers them.

        :param values: The array.
        :type values: np.ndarray
        :param selection: The index or ``None`` per selectable axis; every axis takes
            index ``0`` when omitted.
        :type selection: Mapping[str, int | None] | None
        :returns: The selected k-space, converted by
            ``pypft.acquisition.radial_lines_to_polar``.
        :rtype: RadialKSpace
        :raises TypeError: If any argument has the wrong type.
        :raises ValueError: If the selection names an unknown axis or an index outside
            its axis.
        :raises RadialLineLayoutError: If the array is not in this layout.

        """
        sizes = cls.selectable_axes(values=values)
        index = cls._selection_index(
            sizes=sizes, selection={} if selection is None else selection
        )
        lines = cls._select_lines(values=values, index=index)
        n_readout, n_lines = lines.shape[:_LINES_SAMPLE_NDIM]
        if lines.ndim > _LINES_SAMPLE_NDIM:
            # Every kept axis, flattened into PyPFT's single trailing batch axis.
            lines = lines.reshape((n_readout, n_lines, -1))
        return radial_lines_to_polar(
            lines=lines,
            angles=cls._line_angles(n_lines=n_lines),
            center_index=cls._center_index(n_readout=n_readout),
        )
