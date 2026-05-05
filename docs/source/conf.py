from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

project = "PyPFT"
author = "Hossein Ghasem Damghani"
release = "0.0.5"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

exclude_patterns: list[str] = []

html_theme = "alabaster"
autodoc_member_order = "bysource"
