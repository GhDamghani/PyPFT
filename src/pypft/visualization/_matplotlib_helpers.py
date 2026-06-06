from __future__ import annotations

import sys

import numpy as np

from pypft.core.exceptions import VisualizationDependencyError
from pypft.visualization.arrays import apply_gamma, magnitude, phase
from pypft.visualization.models import RenderView


def load_matplotlib():
    try:
        import matplotlib

        if (
            "matplotlib.pyplot" not in sys.modules
            and matplotlib.get_backend().lower() != "agg"
        ):
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise VisualizationDependencyError(
            "Static figure rendering requires the optional matplotlib "
            "dependency. Install PyPFT with the viz extra."
        ) from error
    return plt


def render_payload(
    data: np.ndarray,
    *,
    view: RenderView,
    gamma: float,
) -> tuple[np.ndarray, str, str, tuple[float, float] | None]:
    if view == "real":
        real_data = np.real(data)
        data_min = float(np.min(real_data))
        data_max = float(np.max(real_data))
        if data_min < 0.0 < data_max:
            limit = max(abs(data_min), abs(data_max))
            return real_data, "RdBu_r", "value", (-limit, limit)
        return real_data, "viridis", "value", None
    if view == "magnitude":
        return (
            apply_gamma(magnitude(data), gamma),
            "viridis",
            "magnitude",
            None,
        )
    return phase(data), "twilight", "phase [rad]", (-np.pi, np.pi)


def axis_labels(frame) -> tuple[str, str]:
    if frame.field_kind == "spatial_samples":
        return ("angular index", "radial index")
    if frame.field_kind == "angular_spectrum":
        return ("angular order", "radial index")
    if frame.field_kind == "radial_spectrum":
        return ("angular order", "radial frequency index")
    return ("angular frequency index", "radial frequency index")


__all__ = ["axis_labels", "load_matplotlib", "render_payload"]
