"""Radial k-space lines, measured or generated, converted into PyPFT's polar data.

Radial MRI records k-space along lines through its center; PyPFT stores polar data as
spokes from the center, ``(radial, angular[, batch])`` with a centered angular axis.
This package bridges the two, in two layers:

- **Layout-independent**: ``radial_lines_to_polar`` splits uniformly spaced lines into
  spokes, and ``RadialKSpace`` carries the result with the frequency radius it covers,
  ready for ``pypft.grid.resample_uniform_polar_frequency`` (the exact path,
  ``pypft.inverse_pft``) or ``pypft.inverse_pft_ring`` (the ring-consistent route).
  Data from any source goes through these two directly.
- **One reader per array layout**: ``read_radial_kspace`` reads a file whose lines
  are laid out as a given ``RadialLineLayout`` describes, whether measured or
  generated, and ``selectable_axes`` lists the axes a selection chooses from (slices,
  phases, coils, ...). A selection takes one index of an axis, or keeps the whole
  axis in the batch.

A new layout is a ``pypft.acquisition._base.BaseRadialReader`` subclass overriding its
four hooks (axis sizes, loading, line angles, center index), one new
``RadialLineLayout`` member, and its ``_READERS`` entry; the selection logic and the
conversion are shared. ``python -m pypft.acquisition`` opens a terminal interface over
the same two functions (``pypft.acquisition.app``).
"""

from collections.abc import Mapping
from enum import Enum, auto
from pathlib import Path

from pypft.utils.validators import EnumValidator

from ._base import BaseRadialReader, RadialLineLayoutError
from ._kspace import NonUniformAnglesError, RadialKSpace, radial_lines_to_polar
from ._slice_phase_readout_line_coil_complex import (
    SlicePhaseReadoutLineCoilComplexReader,
)

__all__ = [
    "RadialLineLayout",
    "RadialLineLayoutError",
    "NonUniformAnglesError",
    "RadialKSpace",
    "layout_suffix",
    "radial_lines_to_polar",
    "read_radial_kspace",
    "selectable_axes",
]


class RadialLineLayout(Enum):
    """The array layouts of radial k-space lines ``read_radial_kspace`` reads.

    Each member names the axes of the stored array in order; its docstring entry
    gives the file it is stored in, the angle of every line and the center of
    k-space.

    - ``SLICE_PHASE_READOUT_LINE_COIL_COMPLEX``: a MATLAB 5 ``.mat`` file with one
      variable, ``data``, of shape ``(slices, phases, readout, lines, coils, 2)``, the
      last axis holding the real and imaginary parts; line ``l`` of ``L`` at
      ``l * pi / L``, the center at readout index ``readout // 2``; selectable axes
      ``slice``, ``phase``, ``coil``. The MD-CNN radial cine dataset (Harvard
      Dataverse, doi:10.7910/DVN/CI3WB6) is stored this way.
    """

    SLICE_PHASE_READOUT_LINE_COIL_COMPLEX = auto()


_READERS: dict[RadialLineLayout, type[BaseRadialReader]] = {
    RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX: (
        SlicePhaseReadoutLineCoilComplexReader
    ),
}


def _reader(layout: RadialLineLayout) -> type[BaseRadialReader]:
    """Validate a layout and return its reader.

    :param layout: The layout.
    :type layout: RadialLineLayout
    :returns: The layout's reader.
    :rtype: type[pypft.acquisition._base.BaseRadialReader]
    :raises TypeError: If ``layout`` is not an enum member.
    :raises ValueError: If it is not an ``RadialLineLayout`` member.

    """
    EnumValidator.type_is_enum(value=layout)
    EnumValidator.value_is_enum_member(value=layout, enum_class=RadialLineLayout)
    return _READERS[layout]


def layout_suffix(layout: RadialLineLayout) -> str:
    """Give the file suffix of a layout, with its leading dot.

    :param layout: The layout.
    :type layout: RadialLineLayout
    :returns: The suffix, e.g. ``".mat"``.
    :rtype: str
    :raises TypeError: If ``layout`` is not an enum member.
    :raises ValueError: If it is not an ``RadialLineLayout`` member.

    """
    return _reader(layout=layout).SUFFIX


def selectable_axes(path: Path, *, layout: RadialLineLayout) -> dict[str, int]:
    """List the axes a selection of a file chooses from, with their sizes.

    Read from the file's header where the layout allows it, without loading its
    data.

    :param path: The file to inspect.
    :type path: Path
    :param layout: The file's layout.
    :type layout: RadialLineLayout
    :returns: Every selectable axis's name and size, in the layout's axis order.
    :rtype: dict[str, int]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``path`` does not have the layout's suffix.
    :raises OSError: If ``path`` cannot be read.
    :raises RadialLineLayoutError: If the file is not laid out as ``layout`` describes.

    """
    return _reader(layout=layout).selectable_axes(path=path)


def read_radial_kspace(
    path: Path,
    *,
    layout: RadialLineLayout,
    selection: Mapping[str, int | None] | None = None,
) -> RadialKSpace:
    """Read a file's radial k-space, as a single sample or a batch of samples.

    ``selection`` maps a selectable axis (``selectable_axes``) to one of its indices,
    or to ``None`` to keep the whole axis; an omitted axis takes index ``0``. With
    every axis at one index the result is a single sample ``(radial, angular)``;
    otherwise every kept axis goes into one batch axis ``(radial, angular, batch)``,
    flattened in C order over the kept axes in the layout's own order, so
    ``values.reshape(radial, angular, *kept_sizes)`` separates them again. For
    example, ``{"phase": 0, "coil": None}`` reads every coil of the first phase as a
    batch.

    :param path: The file to read.
    :type path: Path
    :param layout: The file's layout.
    :type layout: RadialLineLayout
    :param selection: The index, or ``None``, per selectable axis.
    :type selection: Mapping[str, int | None] | None
    :returns: The selected k-space in PyPFT's uniform polar layout.
    :rtype: RadialKSpace
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``path`` does not have the layout's suffix, or the
        selection names an unknown axis or an index outside its axis.
    :raises OSError: If ``path`` cannot be read.
    :raises RadialLineLayoutError: If the file is not laid out as ``layout`` describes.
    :raises NonUniformAnglesError: If the layout's lines are not at uniform angles.

    """
    return _reader(layout=layout).read(path=path, selection=selection)
