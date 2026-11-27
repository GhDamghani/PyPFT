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

# Tracked notebooks are stripped of their outputs, so the docs build executes them
# to render their figures. `cache` re-executes only notebooks whose code changed
# (jupyter-cache, in `docs/.jupyter_cache/`). `09_experimental_data` needs the
# downloaded MD-CNN sample (`python -m pypft.samples --directory tests/samples`).
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True

# Notebook Markdown writes math as `$...$` (inline) and `$$...$$` (display).
myst_enable_extensions = ["dollarmath"]

# A notebook's `[[k]](#ref-k)` citations link to raw-HTML anchors in its own
# References cell, which the browser resolves but MyST cannot see; see
# DESIGN_NOTES.md, "Notebooks: citations link to raw-HTML anchors in a References
# cell". tests/test_notebook_citations.py checks every such link has its anchor.
nitpick_ignore_regex = [("myst", r"ref-\d+")]

# Tracked notebooks carry no metadata at all, so the docs copy is given the one
# key myst-nb needs to pick a syntax highlighter for code cells.
NOTEBOOK_METADATA = {"language_info": {"name": "python"}}

# The copies run from ``docs/_notebooks/``, but their relative paths
# (``../tests/samples``, ``../scripts``) are written for ``notebooks/``. A hidden
# first cell moves the kernel there; ``remove-cell`` keeps it out of the page.
NOTEBOOKS_DIRECTORY = Path(__file__).resolve().parents[1] / "notebooks"
CHDIR_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "id": "docs-chdir",
    "metadata": {"tags": ["remove-cell"]},
    "outputs": [],
    "source": f"import os\nos.chdir({str(NOTEBOOKS_DIRECTORY)!r})",
}


def _copy_notebooks(app, config) -> None:
    """Copy ``notebooks/*.ipynb`` into ``docs/_notebooks/`` before Sphinx reads.

    Sphinx requires every document in a toctree to live under its source
    directory (``docs/``), but ``notebooks/`` is tracked at the repo root as
    one incremental tutorial sequence shared with `nbmake`. Copying at
    ``config-inited`` -- before Sphinx discovers its sources -- makes
    ``tutorials.rst``'s glob toctree see them, without a symlink (which
    needs elevated privileges on Windows). Each copy gains
    ``NOTEBOOK_METADATA``, since the tracked notebooks are stripped of all
    metadata, and ``CHDIR_CELL`` first, so their relative paths resolve.

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx
    :param config: The (already-initialized) Sphinx configuration.
    :type config: sphinx.config.Config

    """
    source = NOTEBOOKS_DIRECTORY
    destination = Path(app.srcdir) / "_notebooks"
    destination.mkdir(exist_ok=True)
    for notebook in source.glob("*.ipynb"):
        content = json.loads(notebook.read_text(encoding="utf-8"))
        content["metadata"] = {**NOTEBOOK_METADATA, **content["metadata"]}
        content["cells"] = [CHDIR_CELL, *content["cells"]]
        (destination / notebook.name).write_text(
            json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8"
        )


def setup(app) -> None:
    """Register the notebook-copying hook (see ``_copy_notebooks``).

    :param app: The running Sphinx application.
    :type app: sphinx.application.Sphinx

    """
    app.connect("config-inited", _copy_notebooks)
