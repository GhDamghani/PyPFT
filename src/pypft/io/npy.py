from __future__ import annotations

from pathlib import Path

import numpy as np

from pypft.core.exceptions import InputShapeError


def load_array(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise InputShapeError("Expected a NumPy array in the .npy input file.")
    return np.asarray(values)


def load_spatial_sample(path: Path) -> np.ndarray:
    values = load_array(path)
    if values.ndim != 2:
        raise InputShapeError(
            "Phase-one CLI workflows require one 2D sample with shape "
            "(n_r, n_theta)."
        )
    return np.asarray(values)


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


__all__ = ["load_array", "load_spatial_sample", "save_array"]
