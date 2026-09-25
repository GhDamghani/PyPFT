Installing PyPFT as a contributor
=================================

This tutorial sets up a development copy of PyPFT, from a fresh clone to a green quality
gate. The project's coding conventions and pull-request process are described in
`CONTRIBUTING.md <https://github.com/GhDamghani/PyPFT/blob/main/CONTRIBUTING.md>`_.

Fork and clone
--------------

Fork `the repository <https://github.com/GhDamghani/PyPFT>`_ on GitHub, then clone your
fork::

   git clone https://github.com/<your-username>/PyPFT.git
   cd PyPFT

Install uv and the environment
------------------------------

PyPFT's dependencies are managed with `uv <https://docs.astral.sh/uv/>`_, and the
lockfile (``uv.lock``) is committed. After installing ``uv``, create the environment,
with both runtime and development dependencies, from the repository root::

   uv sync

Run ``uv sync`` again after every pull or branch switch, so the environment always matches
``pyproject.toml``.

Install the commit hook
-----------------------

Notebooks are committed stripped -- no outputs, no execution counts, no metadata -- and the
test suite checks this. Install the ``pre-commit`` hook that does the stripping for you::

   uv run pre-commit install

Run the quality gate
--------------------

The quality gate is the same script CI runs on Windows, Linux, and macOS. It runs the test
suite, the notebooks, the formatters and linters, the type checker, the dead-code check,
the documentation build, and the package build, stopping at the first failure::

   ./scripts/Invoke-QualityGate.ps1

The script needs PowerShell 7 (``pwsh``), which is available on all three platforms. The
test suite alone runs with::

   uv run pytest

Build the documentation
-----------------------

Warnings fail the build::

   uv run sphinx-build -W docs docs/_build

The rendered pages land in ``docs/_build/``; open ``docs/_build/index.html`` in a browser.

Run a single notebook
---------------------

The tutorial notebooks live in ``notebooks/`` and are executed by ``nbmake`` as part of the
test suite. To execute and check just one of them::

   uv run pytest --nbmake notebooks/00_installation_and_quickstart.ipynb

To edit a notebook interactively, start Jupyter from the project environment::

   uv run jupyter notebook

Where the tests live
--------------------

``tests/`` mirrors the package layout of ``src/pypft/``: the tests for ``src/pypft/dht/``
live in ``tests/dht/``, and the tests for a top-level module such as
``src/pypft/geometry.py`` live directly in ``tests/`` (``tests/test_geometry.py``). Run a
single test with::

   uv run pytest tests/test_geometry.py::<test_name>
