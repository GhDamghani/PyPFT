from __future__ import annotations

from typing import Any

import numpy as np

from pypft.core.typing import ComplexArray


def as_numpy_array(values: Any) -> np.ndarray:
    module_name = type(values).__module__
    if module_name.startswith("cupy"):
        import cupy as cp

        return cp.asnumpy(values)
    return np.asarray(values)


def squeeze_single_sample(values: Any) -> np.ndarray:
    array = as_numpy_array(values)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


def magnitude(values: ComplexArray) -> np.ndarray:
    return np.abs(as_numpy_array(values))


def phase(values: ComplexArray) -> np.ndarray:
    return np.angle(as_numpy_array(values))


def apply_gamma(values: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("Gamma must be positive.")
    if gamma == 1.0:
        return values
    return np.power(np.clip(values, 0.0, None), gamma)


__all__ = [
    "apply_gamma",
    "as_numpy_array",
    "magnitude",
    "phase",
    "squeeze_single_sample",
]
