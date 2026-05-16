from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformConventions:
    spatial_axes: tuple[str, str] = ("radial", "angular")
    frequency_axes: tuple[str, str] = ("radial", "angular")
    batch_axis: str = "leading"
    angular_fft_backend: str = "scipy.fft"
    dht_pairing: str = "shared_forward_backward"
    normalization: str | None = None
    phase_convention: str | None = None
    notes: tuple[str, ...] = (
        "Mathematical normalization and phase conventions remain pending.",
        (
            "Grid coordinate values are intentionally unspecified until the "
            "DHT is implemented."
        ),
        (
            "Only structural axis and dtype contracts are enforced in the "
            "current scaffold."
        ),
    )


DEFAULT_CONVENTIONS = TransformConventions()


__all__ = ["DEFAULT_CONVENTIONS", "TransformConventions"]
