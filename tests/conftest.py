"""Shared pytest configuration and constants.

Every OS in CI (``windows-latest``/``ubuntu-latest``/``macos-latest``) runs headless, with
no display server available for an interactive backend (e.g. ``TkAgg``) to attach to.
``pypft.viz``'s tests are the first in this suite to actually create ``matplotlib``
figures, so the backend must be forced once, before any test module imports
``matplotlib.pyplot``.

``STYLED_NOTEBOOK_PATHS`` is shared by the notebook style and backend-mention lints.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ========================================================================================
# Constants
# ========================================================================================

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

#: The tutorial notebooks the notebook style rules (``test_notebook_style.py``) and the
#: backend-mention rule (``test_no_backend_mentions.py``) are enforced on.
STYLED_NOTEBOOK_NAMES = (
    "00_installation_and_quickstart.ipynb",
    "01_polar_images_and_grids.ipynb",
    "02_sampling_grids.ipynb",
    "03_discrete_hankel_transform.ipynb",
    "04_domains.ipynb",
    "05_pft_and_ipft.ipynb",
    "06_batches.ipynb",
    "07_pft_properties.ipynb",
)
STYLED_NOTEBOOK_PATHS = tuple(NOTEBOOKS_DIR / name for name in STYLED_NOTEBOOK_NAMES)
