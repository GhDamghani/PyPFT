class PyPFTError(Exception):
    """Base exception for PyPFT."""


class InputShapeError(PyPFTError):
    """Raised when an input array shape is incompatible with the API."""


class GridMismatchError(PyPFTError):
    """Raised when an explicit grid does not match the runtime data shape."""


class InvalidFieldOperationError(PyPFTError):
    """Raised when a stage-aware field operation is invalid."""


class UnknownDHTImplementationError(PyPFTError):
    """Raised when the selected DHT implementation key is unavailable."""


class DHTImplementationPendingError(PyPFTError, NotImplementedError):
    """Raised when a placeholder DHT implementation is executed."""


class UnknownBackendError(PyPFTError):
    """Raised when the selected execution backend is unavailable."""


class BackendUnavailableError(PyPFTError):
    """Raised when an optional backend dependency is not installed."""


class MetadataValidationError(PyPFTError):
    """Raised when CLI metadata is missing or scientifically incomplete."""


class VisualizationDependencyError(PyPFTError):
    """Raised when an optional visualization backend dependency is missing."""


__all__ = [
    "BackendUnavailableError",
    "DHTImplementationPendingError",
    "GridMismatchError",
    "InputShapeError",
    "InvalidFieldOperationError",
    "MetadataValidationError",
    "PyPFTError",
    "UnknownBackendError",
    "UnknownDHTImplementationError",
    "VisualizationDependencyError",
]
