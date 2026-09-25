"""Sphinx configuration for PyPFT's documentation."""

import json
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

# Tracked notebooks carry no metadata at all, so the docs copy is given the one
# key myst-nb needs to pick a syntax highlighter for code cells.
NOTEBOOK_METADATA = {"language_info": {"name": "python"}}


def _copy_notebooks(app, config) -> None:
    """Copy ``notebooks/*.ipynb`` into ``docs/_notebooks/`` before Sphinx reads.

    Sphinx requires every document in a toctree to live under its source
    directory (``docs/``), but ``notebooks/`` is tracked at the repo root as
    one incremental tutorial sequence shared with `nbmake`. Copying at
    ``config-inited`` -- before Sphinx discovers its sources -- makes
    ``tutorials.rst``'s glob toctree see them, without a symlink (which
    needs elevated privileges on Windows, one of CI's three platforms). Each
    copy gains ``NOTEBOOK_METADATA``, since the tracked notebooks are stripped
    of all metadata.

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx
    :param config: The (already-initialized) Sphinx configuration.
    :type config: sphinx.config.Config

    """
    source = Path(__file__).resolve().parents[1] / "notebooks"
    destination = Path(app.srcdir) / "_notebooks"
    destination.mkdir(exist_ok=True)
    for notebook in source.glob("*.ipynb"):
        content = json.loads(notebook.read_text(encoding="utf-8"))
        content["metadata"] = {**NOTEBOOK_METADATA, **content["metadata"]}
        (destination / notebook.name).write_text(
            json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8"
        )


def setup(app) -> None:
    """Register the notebook-copying hook (see ``_copy_notebooks``).

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx

    """
    app.connect("config-inited", _copy_notebooks)
