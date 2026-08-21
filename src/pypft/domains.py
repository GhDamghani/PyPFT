"""Typed domain objects: the legal-move shell over the verified PFT numerics.

``forward_pft``/``inverse_pft`` (``pypft.transform``) already compose the angular
DFT/IDFT and the per-harmonic scaled Hankel transform correctly -- this module adds
nothing numerical on top of them. What it adds is a *typed* way to name where a polar
array sits along that chain, and to walk between those points one verified step at a
time:

``SPACE_POLAR --DFT--> SPACE_HARMONIC --DHT--> FREQUENCY_HARMONIC --IDFT-->
FREQUENCY_POLAR``

Word 1 of each ``Domain`` member (``SPACE``/``FREQUENCY``) is the radial coordinate,
changed only by the discrete Hankel transform; word 2 (``POLAR``/``HARMONIC``) is the
angular coordinate, changed only by the angular DFT/IDFT. Because this is a path graph
with no branches, a transition is legal exactly when it moves one step along ``_CHAIN``
-- there is no separate legality table to keep in sync with it.

``values``/``grid``-in, ``values``/``grid``-out stays the primitive: ``BaseSignal`` and
its four subclasses (one per ``Domain`` member) are a thin, optional convenience
wrapping that primitive with its own domain, so the numeric path in ``pypft.transform``
never requires this module.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

import numpy as np

from pypft.axes import Axis
from pypft.dft import angular_dft, inverse_angular_dft
from pypft.grid import PolarGrid, _type_is_polar_grid
from pypft.transform import Direction, scaled_hankel
from pypft.utils.validators import EnumValidator, NumpyValidator

# ======================================================================================
# The domain chain
# ======================================================================================


class Domain(Enum):
    """The four points a polar array occupies across the PFT/IPFT chain.

    Word 1 of each member's name is the radial coordinate (changed only by the
    discrete Hankel transform); word 2 is the angular coordinate (changed only by
    the angular DFT/IDFT) -- see ``_CHAIN``.
    """

    SPACE_POLAR = auto()
    SPACE_HARMONIC = auto()
    FREQUENCY_HARMONIC = auto()
    FREQUENCY_POLAR = auto()


_CHAIN: tuple[Domain, ...] = (
    Domain.SPACE_POLAR,
    Domain.SPACE_HARMONIC,
    Domain.FREQUENCY_HARMONIC,
    Domain.FREQUENCY_POLAR,
)
"""The PFT's single, ordered path of domains, space to frequency. A transition
between two domains is legal exactly when ``abs(i - j) == 1`` over these indices --
there are no branches or cycles, so no separate legal-moves table is needed."""

_STEP_TOWARD: tuple[str, str, str] = ("to_harmonics", "to_frequency", "to_angles")
"""The method that advances a signal from ``_CHAIN[i]`` to ``_CHAIN[i + 1]``, for each
of the chain's three edges -- edges 0 and 2 are angular (DFT), edge 1 is radial (DHT),
matching ``forward_pft``'s own step order."""

_STEP_BACKWARD: tuple[str, str, str] = ("to_angles", "to_space", "to_harmonics")
"""The method that retreats a signal from ``_CHAIN[i + 1]`` to ``_CHAIN[i]``, mirroring
``_STEP_TOWARD`` -- matching ``inverse_pft``'s own step order."""


# ======================================================================================
# The signal value object
# ======================================================================================


@dataclass(frozen=True)
class BaseSignal:
    """A frozen polar array, tagged with the ``Domain`` it currently occupies.

    Every subclass fixes ``domain`` to one ``Domain`` member and defines only the
    step methods for that member's own neighbours in ``_CHAIN`` -- calling a step
    method that does not exist on a given subclass is therefore a ``pyright`` error
    on a hand-written chain, not just a runtime one. ``to`` is the dynamic
    counterpart, walking ``_CHAIN`` to an arbitrary target domain.

    :param values: The signal's samples, on ``grid``'s ``(n_radial, n_angular)``
        layout (``pypft.axes.Axis``).
    :type values: np.ndarray
    :param grid: The sampling grid ``values`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is not 2-D or its shape does not match
        ``grid``.

    """

    values: np.ndarray
    grid: PolarGrid
    domain: ClassVar[Domain]

    def __post_init__(self) -> None:
        """Validate ``values``/``grid`` once, right after construction."""
        NumpyValidator.type_is_ndarray(value=self.values)
        NumpyValidator.value_is_2d(value=self.values)
        _type_is_polar_grid(value=self.grid)
        reference = np.empty((self.grid.n_radial, self.grid.n_angular))
        NumpyValidator.value1_shape_matches_value2(value1=self.values, value2=reference)

    def to(self, domain: Domain) -> "BaseSignal":
        """Walk ``_CHAIN`` from this signal's own domain to ``domain``.

        A step at a time along the single ordered chain -- never a general graph
        search -- since the only decision at each step is which direction to walk
        and which named method (``_STEP_TOWARD``/``_STEP_BACKWARD``) advances one
        edge in that direction.

        :param domain: The domain to walk to.
        :type domain: Domain
        :returns: This signal transformed into ``domain``.
        :rtype: BaseSignal
        :raises TypeError: If ``domain`` is not a ``Domain``.
        :raises ValueError: If ``domain`` is not a ``Domain`` member.

        """
        EnumValidator.type_is_enum(value=domain)
        EnumValidator.value_is_enum_member(value=domain, enum_class=Domain)
        start, end = _CHAIN.index(self.domain), _CHAIN.index(domain)
        step = 1 if end >= start else -1
        signal: BaseSignal = self
        for edge in range(start, end, step):
            method = _STEP_TOWARD[edge] if step == 1 else _STEP_BACKWARD[edge - 1]
            signal = getattr(signal, method)()
        return signal


@dataclass(frozen=True)
class SpacePolarSignal(BaseSignal):
    """The space domain on the physical angle axis: ``f(r, theta)``."""

    domain: ClassVar[Domain] = Domain.SPACE_POLAR

    def to_harmonics(self) -> "SpaceHarmonicSignal":
        """Apply the angular DFT, moving to the space domain's harmonic axis.

        :returns: The equivalent signal in ``Domain.SPACE_HARMONIC``.
        :rtype: SpaceHarmonicSignal

        """
        values = angular_dft(x=self.values, axis=Axis.ANGULAR)
        return SpaceHarmonicSignal(values=values, grid=self.grid)


@dataclass(frozen=True)
class SpaceHarmonicSignal(BaseSignal):
    """The space domain on the harmonic-order axis: ``f_n(r)``."""

    domain: ClassVar[Domain] = Domain.SPACE_HARMONIC

    def to_angles(self) -> SpacePolarSignal:
        """Apply the angular IDFT, moving back to the physical angle axis.

        :returns: The equivalent signal in ``Domain.SPACE_POLAR``.
        :rtype: SpacePolarSignal

        """
        values = inverse_angular_dft(X=self.values, axis=Axis.ANGULAR)
        return SpacePolarSignal(values=values, grid=self.grid)

    def to_frequency(self) -> "FrequencyHarmonicSignal":
        """Apply the scaled forward Hankel transform, moving to the frequency domain.

        :returns: The equivalent signal in ``Domain.FREQUENCY_HARMONIC``.
        :rtype: FrequencyHarmonicSignal

        """
        values = scaled_hankel(
            values=self.values,
            grid=self.grid,
            direction=Direction.FORWARD,
            axis=Axis.RADIAL,
        )
        return FrequencyHarmonicSignal(values=values, grid=self.grid)


@dataclass(frozen=True)
class FrequencyHarmonicSignal(BaseSignal):
    """The frequency domain on the harmonic-order axis: ``F_n(rho)``."""

    domain: ClassVar[Domain] = Domain.FREQUENCY_HARMONIC

    def to_space(self) -> SpaceHarmonicSignal:
        """Apply the scaled inverse Hankel transform, moving back to the space domain.

        :returns: The equivalent signal in ``Domain.SPACE_HARMONIC``.
        :rtype: SpaceHarmonicSignal

        """
        values = scaled_hankel(
            values=self.values,
            grid=self.grid,
            direction=Direction.INVERSE,
            axis=Axis.RADIAL,
        )
        return SpaceHarmonicSignal(values=values, grid=self.grid)

    def to_angles(self) -> "FrequencyPolarSignal":
        """Apply the angular IDFT, moving to the frequency domain's angle axis.

        :returns: The equivalent signal in ``Domain.FREQUENCY_POLAR``.
        :rtype: FrequencyPolarSignal

        """
        values = inverse_angular_dft(X=self.values, axis=Axis.ANGULAR)
        return FrequencyPolarSignal(values=values, grid=self.grid)


@dataclass(frozen=True)
class FrequencyPolarSignal(BaseSignal):
    """The frequency domain on the physical angle axis: ``F(rho, phi)``."""

    domain: ClassVar[Domain] = Domain.FREQUENCY_POLAR

    def to_harmonics(self) -> FrequencyHarmonicSignal:
        """Apply the angular DFT, moving back to the harmonic-order axis.

        :returns: The equivalent signal in ``Domain.FREQUENCY_HARMONIC``.
        :rtype: FrequencyHarmonicSignal

        """
        values = angular_dft(x=self.values, axis=Axis.ANGULAR)
        return FrequencyHarmonicSignal(values=values, grid=self.grid)
