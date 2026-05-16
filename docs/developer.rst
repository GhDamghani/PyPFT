Developer Guide
===============

Environment setup
-----------------

Use the installer script from the repository root:

.. code-block:: bash

   python scripts/install_dev.py

That creates ``.venv`` and installs the package in editable mode with development, documentation, and benchmark extras.

To include the optional CUDA 12.x CuPy backend:

.. code-block:: bash

   python scripts/install_dev.py --gpu

Core commands
-------------

Run tests:

.. code-block:: bash

   python -m pytest

GPU-specific tests are guarded with skip markers and only run when CuPy is
installed and importable in the current environment.

Build the docs:

.. code-block:: bash

   python -m sphinx -b html docs docs/build/html

Build source and wheel distributions:

.. code-block:: bash

   python -m build

Release model
-------------

The repository uses GitHub Actions workflows to validate tests and docs on push and pull request. Tagged releases build distributions, publish release artifacts to GitHub, and publish to PyPI through Trusted Publishing.

GPU notes
---------

The main CI remains CPU-only because GitHub-hosted runners do not provide CUDA
hardware. The current GPU milestone should be verified locally on a CUDA 12.x
machine or on a future self-hosted GPU runner.

