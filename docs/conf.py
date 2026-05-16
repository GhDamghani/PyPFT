from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "PyPFT"
author = "GhDamghani"
copyright = "2026, GhDamghani"
release = "0.1.0.dev0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_static_path: list[str] = []
