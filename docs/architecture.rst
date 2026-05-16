Architecture
============

Public API
----------

PyPFT exposes a single facade object with a small constructor surface and two main methods:

- ``PyPFT.forward(image)``
- ``PyPFT.backward(image)``

Internal layers
---------------

The package is organized so each transform stage can evolve independently.

- ``pypft.api`` contains the public ``PyPFT`` facade.
- ``pypft.backends`` owns array conversion and angular FFT/IFFT dispatch.
- ``pypft.config`` holds immutable user-facing configuration.
- ``pypft.core`` contains conventions, exceptions, typing aliases, and input normalization.
- ``pypft.grids`` contains structural spatial and frequency grid objects.
- ``pypft.dft`` and ``pypft.idft`` wrap the backend-driven angular FFT and IFFT stages.
- ``pypft.dht`` contains the DHT registry and the current mock implementation.
- ``pypft.operators`` contains forward and backward plan orchestration.
- ``pypft.validation`` and ``pypft.visualization`` reserve space for reference cases and array inspection helpers.

Execution flow
--------------

The current execution path is:

1. Validate the input shape and coerce it to ``complex128``.
2. Normalize the internal shape to ``(batch, n_r, n_theta)``.
3. Apply the angular FFT along the last axis through the selected backend.
4. Dispatch to the selected DHT implementation key.
5. Apply the angular IFFT after the radial stage through the selected backend.

The current DHT key is ``mock-mirror``, which returns its input unchanged.
That gives the package a concrete end-to-end execution path while preserving the
same operator boundaries needed for a future real DHT implementation.

The current backend options are ``cpu`` and ``gpu``. CPU remains the default.
GPU requires the optional CuPy extra and currently accelerates only the array
and angular FFT/IFFT path, not a real radial DHT kernel.
