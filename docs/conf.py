"""Sphinx configuration for PyPFT's documentation."""

import shutil
from pathlib import Path

project = "PyPFT"
copyright = "2026, GhDamghani"
author = "GhDamghani"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_nb",
]

exclude_patterns = ["_build", "jupyter_execute", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

# Notebooks are executed and checked by `nbmake` in CI (scripts/Test-Notebooks.ps1),
# not by Sphinx -- building the docs only renders their already-committed output.
nb_execution_mode = "off"


def _copy_notebooks(app, config) -> None:
    """Copy ``notebooks/*.ipynb`` into ``docs/_notebooks/`` before Sphinx reads.

    Sphinx requires every document in a toctree to live under its source
    directory (``docs/``), but ``notebooks/`` is tracked at the repo root as
    one incremental tutorial sequence shared with `nbmake`. Copying at
    ``config-inited`` -- before Sphinx discovers its sources -- makes
    ``tutorials.rst``'s glob toctree see them, without a symlink (which
    needs elevated privileges on Windows, one of CI's three platforms).

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx
    :param config: The (already-initialized) Sphinx configuration.
    :type config: sphinx.config.Config

    """
    source = Path(__file__).resolve().parents[1] / "notebooks"
    destination = Path(app.srcdir) / "_notebooks"
    destination.mkdir(exist_ok=True)
    for notebook in source.glob("*.ipynb"):
        shutil.copyfile(notebook, destination / notebook.name)


def setup(app) -> None:
    """Register the notebook-copying hook (see ``_copy_notebooks``).

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx

    """
    app.connect("config-inited", _copy_notebooks)
