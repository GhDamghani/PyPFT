GPU Backend
===========

PyPFT supports an optional CuPy-backed GPU execution path.

Installation
------------

Install the GPU extra on a CUDA 12.x machine:

.. code-block:: bash

   python -m pip install -e .[gpu]

For a full local development environment with GPU support:

.. code-block:: bash

   python scripts/install_dev.py --gpu

Usage
-----

Select the GPU backend explicitly when constructing the facade:

.. code-block:: python

   from pypft import PyPFT

   pft = PyPFT(backend="gpu")

Current scope
-------------

The current GPU milestone is intentionally narrow.

- Input arrays are converted to CuPy ``complex128`` arrays.
- Angular FFT and IFFT stages dispatch through CuPy.
- The current compatibility ``naive`` DHT keeps the same end-to-end API path on GPU.

Current limitations
-------------------

- Real GPU-accelerated DHT kernels are not implemented yet; the current direct reference DHT runs through CPU-side NumPy/SciPy math and converts the result back to CuPy.
- Visualization helpers still assume CPU-oriented array consumers.
- The main CI does not execute GPU tests because GitHub-hosted runners lack CUDA hardware.

Validation
----------

When CuPy is installed, the test suite includes GPU-marked parity tests that
compare the current compatibility-key GPU path against the CPU path for small arrays.
