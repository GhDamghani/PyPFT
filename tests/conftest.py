"""Shared pytest configuration: force a non-interactive matplotlib backend.

Every OS in CI (``windows-latest``/``ubuntu-latest``/``macos-latest``) runs headless, with
no display server available for an interactive backend (e.g. ``TkAgg``) to attach to.
``pypft.viz``'s tests are the first in this suite to actually create ``matplotlib``
figures, so the backend must be forced once, before any test module imports
``matplotlib.pyplot``.
"""

import matplotlib

matplotlib.use("Agg")
