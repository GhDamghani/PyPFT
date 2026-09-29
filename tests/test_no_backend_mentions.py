"""User-facing prose never names the image-resampling backend.

Tutorials and top-level documentation describe PyPFT's own conventions only, never
a backend library's: its name, import name, or function names. Every cell of a
styled notebook (Markdown and code alike) is checked, along with ``README.md``,
``CONTRIBUTING.md``, and the hand-written ``docs/*.rst`` pages. The dependency's
own distribution name in an install command is not a mention of its API and is
not flagged.
"""

import json
import re
from pathlib import Path

import pytest

from .conftest import STYLED_NOTEBOOK_PATHS

# ========================================================================================
# Constants
# ========================================================================================

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROSE_PATHS = (
    _REPO_ROOT / "README.md",
    _REPO_ROOT / "CONTRIBUTING.md",
    *sorted((_REPO_ROOT / "docs").glob(pattern="*.rst")),
)
_BACKEND_PATTERN = re.compile(
    r"\bcv2\b"  # the import name
    r"|\bcv_"  # a variable prefix named after the backend
    r"|\bcv\b"  # the usual import alias
    r"|(?i:opencv)(?!-python)"  # the library name, but not its distribution name
    r"|\bwarpPolar\b"
    r"|\bremap\b"
)


# ========================================================================================
# Helpers
# ========================================================================================


def _notebook_text(path: Path) -> str:
    """Join every cell's source of one notebook.

    :param path: The ``.ipynb`` file to read.
    :type path: Path
    :returns: All cell sources, newline-joined.
    :rtype: str

    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def _mentions(text: str) -> list[str]:
    """List every backend mention in ``text``.

    :param text: The text to scan.
    :type text: str
    :returns: Each matched mention, in order.
    :rtype: list[str]

    """
    return [match.group(0) for match in _BACKEND_PATTERN.finditer(text)]


# ========================================================================================
# Tests
# ========================================================================================


@pytest.mark.parametrize(
    argnames="path", argvalues=STYLED_NOTEBOOK_PATHS, ids=lambda p: p.name
)
def test_notebook_does_not_mention_the_backend(path: Path) -> None:
    """No cell of a styled notebook names the backend."""
    mentions = _mentions(text=_notebook_text(path=path))
    assert not mentions, f"{path.name} names the backend: {mentions}"


@pytest.mark.parametrize(
    argnames="path",
    argvalues=_PROSE_PATHS,
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_documentation_does_not_mention_the_backend(path: Path) -> None:
    """No top-level documentation page names the backend."""
    mentions = _mentions(text=path.read_text(encoding="utf-8"))
    assert not mentions, f"{path.name} names the backend: {mentions}"


@pytest.mark.parametrize(
    argnames=("text", "expected"),
    argvalues=[
        ("import cv2", ["cv2"]),
        ("import cv2 as cv", ["cv2", "cv"]),
        ("cv_angle = 0.0", ["cv_"]),
        ("wraps OpenCV", ["OpenCV"]),
        ("calls warpPolar", ["warpPolar"]),
        ("one remap call", ["remap"]),
        ("pip install opencv-python-headless", []),
        ("a curve and its csv file", []),
    ],
)
def test_backend_pattern(text: str, expected: list[str]) -> None:
    """The pattern catches every backend spelling and nothing else."""
    assert _mentions(text=text) == expected
