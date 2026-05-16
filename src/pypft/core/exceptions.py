class PyPFTError(Exception):
    """Base exception for PyPFT."""


class InputShapeError(PyPFTError):
    """Raised when an input array shape is incompatible with the API."""


class GridMismatchError(PyPFTError):
    """Raised when an explicit grid does not match the runtime data shape."""


class UnknownDHTImplementationError(PyPFTError):
    """Raised when the selected DHT implementation key is unavailable."""


class DHTImplementationPendingError(PyPFTError, NotImplementedError):
    """Raised when a placeholder DHT implementation is executed."""


class UnknownBackendError(PyPFTError):
    """Raised when the selected execution backend is unavailable."""


class BackendUnavailableError(PyPFTError):
    """Raised when an optional backend dependency is not installed."""


__all__ = [
    "BackendUnavailableError",
    "DHTImplementationPendingError",
    "GridMismatchError",
    "InputShapeError",
    "PyPFTError",
    "UnknownBackendError",
    "UnknownDHTImplementationError",
]
