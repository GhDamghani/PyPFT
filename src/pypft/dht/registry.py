from __future__ import annotations

from typing import Final

from pypft.core.exceptions import UnknownDHTImplementationError
from pypft.dht.base import DHTImplementation
from pypft.dht.mock_mirror import MockMirrorDHTImplementation

_REGISTRY: Final[dict[str, DHTImplementation]] = {
    "mock-mirror": MockMirrorDHTImplementation(),
}


def available_dht_implementations() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def create_dht_implementation(key: str) -> DHTImplementation:
    try:
        return _REGISTRY[key]
    except KeyError as error:
        available = ", ".join(available_dht_implementations())
        raise UnknownDHTImplementationError(
            f"Unknown DHT implementation {key!r}. Available "
            f"implementations: {available}."
        ) from error


__all__ = ["available_dht_implementations", "create_dht_implementation"]
