Quickstart
==========

The public entry point is the ``PyPFT`` object.

.. code-block:: python

   import numpy as np
   from pypft import PyPFT

   image = np.ones((16, 32), dtype=np.complex128)
   pft = PyPFT()
    transformed = pft.forward(image)
    reconstructed = pft.backward(transformed)

The current default DHT implementation is ``mock-mirror``. It returns the
radial-stage input unchanged, so the existing forward and backward paths act as
identity transforms while the real DHT implementation is still under design.

Use ``enable_batching=True`` to accept a leading batch axis:

.. code-block:: python

   batch = np.ones((4, 16, 32), dtype=np.complex128)
   pft = PyPFT(enable_batching=True)
    transformed_batch = pft.forward(batch)

The batch path is validated and wired through the same operator plans, and the
current ``mock-mirror`` DHT returns the batched values unchanged.

Optional GPU execution is available when the CuPy extra is installed:

.. code-block:: python

    pft = PyPFT(backend="gpu")
    transformed_gpu = pft.forward(image)

The current GPU milestone accelerates the backend array path and angular FFT
stages only. The radial stage still uses the same ``mock-mirror`` DHT contract.
