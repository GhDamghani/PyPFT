"""Shared pytest configuration and constants.

Every OS in CI (``windows-latest``/``ubuntu-latest``/``macos-latest``) runs headless, with
no display server available for an interactive backend (e.g. ``TkAgg``) to attach to.
``pypft.viz``'s tests are the first in this suite to actually create ``matplotlib``
figures, so the backend must be forced once, before any test module imports
``matplotlib.pyplot``.

``NOTEBOOK_PATHS`` is shared by the notebook style and backend-mention lints.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ========================================================================================
# Constants
# ========================================================================================

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

#: Every tutorial notebook: the notebook style rules (``test_notebook_style.py``) and
#: the backend-mention rule (``test_no_backend_mentions.py``) are enforced on each.
NOTEBOOK_PATHS = tuple(sorted(NOTEBOOKS_DIR.glob(pattern="*.ipynb")))
