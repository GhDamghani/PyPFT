"""Tests for DHT implementation selection and registry behavior.

This module verifies only the string-key lookup layer, not radial transform
math. It checks the exposed implementation keys, successful resolution of a
known key, and the failure path for missing implementations.
"""

from __future__ import annotations

import pytest

from pypft.core import UnknownDHTImplementationError
from pypft.dht import available_dht_implementations, create_dht_implementation


def test_available_dht_implementations_are_sorted() -> None:
    assert available_dht_implementations() == ("naive",)


def test_create_known_dht_implementation() -> None:
    implementation = create_dht_implementation("naive")

    assert implementation.key == "naive"
    assert implementation.supports_batching is True


def test_unknown_dht_key_raises() -> None:
    with pytest.raises(UnknownDHTImplementationError):
        create_dht_implementation("missing")
