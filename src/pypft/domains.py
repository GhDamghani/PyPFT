"""Typed domain objects: the legal-move shell over the verified PFT numerics.

``forward_pft``/``inverse_pft`` (``pypft.transform``) already compose the angular
DFT/IDFT and the per-harmonic scaled Hankel transform correctly -- this module adds
nothing numerical on top of them. What it adds is a *typed* way to name where a polar
array sits along that chain, and to walk between those points one verified step at a
time:

``SPATIAL_ANGULAR --DFT--> SPATIAL_HARMONIC --DHT--> FREQUENCY_HARMONIC
--IDFT--> FREQUENCY_ANGULAR``

The naming rule: each coordinate system has its own domain enum (``PolarDomain``
here), and a member names the state of every coordinate group of that system, with
no system prefix. For a polar sample the groups are the radial coordinate
(``SPATIAL``/``FREQUENCY``, changed only by the discrete Hankel transform) and the
angular coordinate (``ANGULAR`` for a physical angle, ``HARMONIC`` for a harmonic
order, changed only by the angular DFT/IDFT). Step methods are named after the
domain they move *into* (``to_spatial_harmonic``, ...), so a hand-written chain reads
as the chain itself; the signal's own class (``PolarSpatialAngularSignal``, ...)
already names the system. Because this is a path graph with no branches, a
transition is legal exactly when it moves one step along ``_POLAR_CHAIN`` -- there is
no separate legality table to keep in sync with it. See ``DESIGN_NOTES.md``,
"Domains: one state word per coordinate group."

``values``/``grid``-in, ``values``/``grid``-out stays the primitive: ``BaseSignal`` and
its four subclasses (one per ``PolarDomain`` member) are a thin, optional convenience
wrapping that primitive with its own domain, so the numeric path in ``pypft.transform``
never requires this module.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

import numpy as np
from matplotlib.axes import Axes

from pypft.axes import DEFAULT_BATCH_AXIS, PolarAxis
from pypft.dft import angular_dft, inverse_angular_dft
from pypft.grid import PolarGrid
from pypft.transform import Direction, _validate_pft_input, scaled_hankel
from pypft.utils.validators import EnumValidator

# ======================================================================================
# The domain chain
# ======================================================================================


class PolarDomain(Enum):
    """The four points a polar array occupies across the PFT/IPFT chain.

    The first word (``SPATIAL``/``FREQUENCY``) is the radial coordinate's state
    (changed only by the discrete Hankel transform); the second (``ANGULAR``/
    ``HARMONIC``) is the angular coordinate's: a physical angle or a harmonic order
    (changed only by the angular DFT/IDFT) -- see ``_POLAR_CHAIN``.
    """

    SPATIAL_ANGULAR = auto()
    SPATIAL_HARMONIC = auto()
    FREQUENCY_HARMONIC = auto()
    FREQUENCY_ANGULAR = auto()


_POLAR_CHAIN: tuple[PolarDomain, ...] = (
    PolarDomain.SPATIAL_ANGULAR,
    PolarDomain.SPATIAL_HARMONIC,
    PolarDomain.FREQUENCY_HARMONIC,
    PolarDomain.FREQUENCY_ANGULAR,
)
"""The PFT's single, ordered path of domains, space to frequency. A transition
between two domains is legal exactly when ``abs(i - j) == 1`` over these indices --
there are no branches or cycles, so no separate legal-moves table is needed."""

_STEP_TOWARD: tuple[str, str, str] = (
    "to_spatial_harmonic",
    "to_frequency_harmonic",
    "to_frequency_angular",
)
"""The method that advances a signal from ``_POLAR_CHAIN[i]`` to
``_POLAR_CHAIN[i + 1]``, for each of the chain's three edges -- edges 0 and 2 are
angular (DFT), edge 1 is radial (DHT), matching ``forward_pft``'s own step order."""

_STEP_BACKWARD: tuple[str, str, str] = (
    "to_spatial_angular",
    "to_spatial_harmonic",
    "to_frequency_harmonic",
)
"""The method that retreats a signal from ``_POLAR_CHAIN[i + 1]`` to
``_POLAR_CHAIN[i]``, mirroring ``_STEP_TOWARD`` -- matching ``inverse_pft``'s own step
order."""


# ======================================================================================
# The signal value object
# ======================================================================================


@dataclass(frozen=True)
class BaseSignal:
    """A frozen polar array, tagged with the ``PolarDomain`` it currently occupies.

    Every subclass fixes ``domain`` to one ``PolarDomain`` member and defines only the
    step methods for that member's own neighbours in ``_POLAR_CHAIN`` -- calling a step
    method that does not exist on a given subclass is therefore a ``pyright`` error
    on a hand-written chain, not just a runtime one. ``to`` is the dynamic
    counterpart, walking ``_POLAR_CHAIN`` to an arbitrary target domain.

    :param values: The signal's samples: a single sample on ``grid``'s
        ``(n_radial, n_angular)`` layout (``pypft.axes.PolarAxis``), or a batch
        ``(n_radial, n_angular, batch)`` of them.
    :type values: np.ndarray
    :param grid: The sampling grid ``values`` is defined on.
    :type grid: pypft.grid.PolarGrid
    :param batch_axis: The axis of ``values`` holding the batch dimension, only
        meaningful for a batch -- PyPFT's own layout always places it after the
        sample axes, so the only accepted value is
        ``pypft.axes.DEFAULT_BATCH_AXIS``.
    :type batch_axis: int
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``values`` is neither a single sample nor a batch, or
        its shape does not match ``grid``/``batch_axis``.

    """

    values: np.ndarray
    grid: PolarGrid
    domain: ClassVar[PolarDomain]
    batch_axis: int = DEFAULT_BATCH_AXIS

    def __post_init__(self) -> None:
        """Validate ``values``/``grid``/``batch_axis``, right after construction."""
        _validate_pft_input(
            values=self.values, grid=self.grid, batch_axis=self.batch_axis
        )

    def to(self, domain: PolarDomain) -> "BaseSignal":
        """Walk ``_POLAR_CHAIN`` from this signal's own domain to ``domain``.

        A step at a time along the single ordered chain -- never a general graph
        search -- since the only decision at each step is which direction to walk
        and which named method (``_STEP_TOWARD``/``_STEP_BACKWARD``) advances one
        edge in that direction.

        :param domain: The domain to walk to.
        :type domain: PolarDomain
        :returns: This signal transformed into ``domain``.
        :rtype: BaseSignal
        :raises TypeError: If ``domain`` is not a ``PolarDomain``.
        :raises ValueError: If ``domain`` is not a ``PolarDomain`` member.

        """
        EnumValidator.type_is_enum(value=domain)
        EnumValidator.value_is_enum_member(value=domain, enum_class=PolarDomain)
        start, end = _POLAR_CHAIN.index(self.domain), _POLAR_CHAIN.index(domain)
        step = 1 if end >= start else -1
        signal: BaseSignal = self
        for edge in range(start, end, step):
            method = _STEP_TOWARD[edge] if step == 1 else _STEP_BACKWARD[edge - 1]
            signal = getattr(signal, method)()
        return signal

    def plot(self, ax: tuple[Axes, Axes] | None = None) -> tuple[Axes, Axes]:
        """Delegate to ``pypft.viz.plot_signal`` for this signal.

        :param ax: See ``pypft.viz.plot_signal``.
        :type ax: tuple[Axes, Axes] | None
        :returns: See ``pypft.viz.plot_signal``.
        :rtype: tuple[Axes, Axes]

        """
        # Deferred import: pypft.viz imports PolarDomain/BaseSignal from this
        # module, so importing it at module level here would be circular.
        from pypft.viz import plot_signal

        return plot_signal(signal=self, ax=ax)


def _type_is_base_signal(value: BaseSignal) -> None:
    """Type-validator for ``BaseSignal``, defined here since the type is defined here.

    :param value: The value to be validated.
    :type value: BaseSignal
    :raises TypeError: If the value is not a ``BaseSignal``.

    """
    if not isinstance(value, BaseSignal):
        raise TypeError(f"value must be BaseSignal, got {type(value).__name__}")


@dataclass(frozen=True)
class PolarSpatialAngularSignal(BaseSignal):
    """The spatial domain on the physical angle axis: ``f(r, theta)``."""

    domain: ClassVar[PolarDomain] = PolarDomain.SPATIAL_ANGULAR

    def to_spatial_harmonic(self) -> "PolarSpatialHarmonicSignal":
        """Apply the angular DFT, moving to the spatial domain's harmonic axis.

        :returns: The equivalent signal in ``PolarDomain.SPATIAL_HARMONIC``.
        :rtype: PolarSpatialHarmonicSignal

        """
        values = angular_dft(x=self.values, axis=PolarAxis.ANGULAR)
        return PolarSpatialHarmonicSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )


@dataclass(frozen=True)
class PolarSpatialHarmonicSignal(BaseSignal):
    """The spatial domain on the harmonic-order axis: ``f_n(r)``."""

    domain: ClassVar[PolarDomain] = PolarDomain.SPATIAL_HARMONIC

    def to_spatial_angular(self) -> PolarSpatialAngularSignal:
        """Apply the angular IDFT, moving back to the physical angle axis.

        :returns: The equivalent signal in ``PolarDomain.SPATIAL_ANGULAR``.
        :rtype: PolarSpatialAngularSignal

        """
        values = inverse_angular_dft(X=self.values, axis=PolarAxis.ANGULAR)
        return PolarSpatialAngularSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )

    def to_frequency_harmonic(self) -> "PolarFrequencyHarmonicSignal":
        """Apply the scaled forward Hankel transform, moving to the frequency domain.

        :returns: The equivalent signal in ``PolarDomain.FREQUENCY_HARMONIC``.
        :rtype: PolarFrequencyHarmonicSignal

        """
        values = scaled_hankel(
            values=self.values,
            grid=self.grid,
            direction=Direction.FORWARD,
            axis=PolarAxis.RADIAL,
            angular_axis=PolarAxis.ANGULAR,
        )
        return PolarFrequencyHarmonicSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )


@dataclass(frozen=True)
class PolarFrequencyHarmonicSignal(BaseSignal):
    """The frequency domain on the harmonic-order axis: ``F_n(rho)``."""

    domain: ClassVar[PolarDomain] = PolarDomain.FREQUENCY_HARMONIC

    def to_spatial_harmonic(self) -> PolarSpatialHarmonicSignal:
        """Apply the scaled inverse Hankel transform, moving back to the spatial domain.

        :returns: The equivalent signal in ``PolarDomain.SPATIAL_HARMONIC``.
        :rtype: PolarSpatialHarmonicSignal

        """
        values = scaled_hankel(
            values=self.values,
            grid=self.grid,
            direction=Direction.INVERSE,
            axis=PolarAxis.RADIAL,
            angular_axis=PolarAxis.ANGULAR,
        )
        return PolarSpatialHarmonicSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )

    def to_frequency_angular(self) -> "PolarFrequencyAngularSignal":
        """Apply the angular IDFT, moving to the frequency domain's angle axis.

        :returns: The equivalent signal in ``PolarDomain.FREQUENCY_ANGULAR``.
        :rtype: PolarFrequencyAngularSignal

        """
        values = inverse_angular_dft(X=self.values, axis=PolarAxis.ANGULAR)
        return PolarFrequencyAngularSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )


@dataclass(frozen=True)
class PolarFrequencyAngularSignal(BaseSignal):
    """The frequency domain on the physical angle axis: ``F(rho, phi)``."""

    domain: ClassVar[PolarDomain] = PolarDomain.FREQUENCY_ANGULAR

    def to_frequency_harmonic(self) -> PolarFrequencyHarmonicSignal:
        """Apply the angular DFT, moving back to the harmonic-order axis.

        :returns: The equivalent signal in ``PolarDomain.FREQUENCY_HARMONIC``.
        :rtype: PolarFrequencyHarmonicSignal

        """
        values = angular_dft(x=self.values, axis=PolarAxis.ANGULAR)
        return PolarFrequencyHarmonicSignal(
            values=values, grid=self.grid, batch_axis=self.batch_axis
        )
