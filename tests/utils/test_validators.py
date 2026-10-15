"""Tests for the shared validators in ``pypft.utils.validators``."""

from collections import OrderedDict
from pathlib import Path

import pytest

from pypft.utils.validators import (
    IntValidator,
    MappingValidator,
    PathValidator,
    StrValidator,
)


@pytest.mark.parametrize(("value1", "value2"), [(0, 1), (-3, 0), (4, 5)])
def test_value1_is_less_than_value2_accepts_smaller(value1: int, value2: int) -> None:
    IntValidator.value1_is_less_than_value2(value1=value1, value2=value2)


@pytest.mark.parametrize(("value1", "value2"), [(1, 1), (5, 4)])
def test_value1_is_less_than_value2_rejects_equal_or_larger(
    value1: int, value2: int
) -> None:
    with pytest.raises(ValueError, match="less than"):
        IntValidator.value1_is_less_than_value2(value1=value1, value2=value2)


def test_type_is_str() -> None:
    StrValidator.type_is_str(value="coil")
    with pytest.raises(TypeError, match="str"):
        StrValidator.type_is_str(value=b"coil")  # type: ignore[arg-type]


#: The suffixes the suffix-set tests accept.
_SUFFIXES = (".npy", ".tif", ".png")


@pytest.mark.parametrize("name", ["a.npy", "b.TIF", "dir/c.png"])
def test_value_has_suffix_in_accepts_listed_suffixes(name: str) -> None:
    PathValidator.value_has_suffix_in(value=Path(name), suffixes=_SUFFIXES)


@pytest.mark.parametrize("name", ["a.npz", "b", "c.tif.gz"])
def test_value_has_suffix_in_rejects_other_suffixes(name: str) -> None:
    with pytest.raises(ValueError, match="one of the suffixes"):
        PathValidator.value_has_suffix_in(value=Path(name), suffixes=_SUFFIXES)


@pytest.mark.parametrize("value", [{}, {"coil": 0}, OrderedDict(phase=None)])
def test_type_is_mapping_accepts_mappings(value: object) -> None:
    MappingValidator.type_is_mapping(value=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[("coil", 0)], "coil", None])
def test_type_is_mapping_rejects_non_mappings(value: object) -> None:
    with pytest.raises(TypeError, match="Mapping"):
        MappingValidator.type_is_mapping(value=value)  # type: ignore[arg-type]
