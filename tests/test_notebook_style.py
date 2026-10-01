"""Machine-checks the style rules of the tutorial notebooks.

- **No hard-wrapped Markdown.** Every paragraph and every list item is one source
  line, so a notebook reads the same whatever width it is displayed at. Two
  consecutive prose lines, or a list item followed by a prose line, mean a
  paragraph has been wrapped by hand. Headings, list items, tables, fenced code,
  and display math are line-structured on purpose and are never flagged.
- **Math is rendered, never typed as code.** Backticks mark Python identifiers
  only; a backticked span containing ``^`` or ``_{`` is math that belongs between
  ``$...$`` instead.
- **Figures are drawn inline.** A notebook that draws figures starts its first code
  cell with ``%matplotlib inline`` (after any comment lines), whatever backend the
  viewer's kernel would otherwise default to.
"""

import json
import re
from pathlib import Path

import pytest

from .conftest import NOTEBOOK_PATHS

# ========================================================================================
# Constants
# ========================================================================================

_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
_DISPLAY_MATH_DELIMITER = "$$"
_HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s")
_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_TABLE_ROW_PREFIX = "|"
# A single-backtick inline code span (double-backtick spans are not used in notebooks).
_CODE_SPAN_PATTERN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_MATH_IN_CODE_MARKERS = ("^", "_{")
_INLINE_MAGIC = "%matplotlib inline"
_COMMENT_PREFIX = "#"
# Any of these in a code cell means the notebook draws figures.
_PLOTTING_PATTERN = re.compile(
    r"\bimport matplotlib\b|\bfrom matplotlib\b|\bplot_signal\(|\.plot\("
    r"|\brender_cartesian\(|\bvisualize_(?:steps|pipeline)\s*=\s*True"
)


# ========================================================================================
# Helpers
# ========================================================================================


def _markdown_sources(path: Path) -> list[str]:
    """Collect the source of every Markdown cell of one notebook.

    :param path: The ``.ipynb`` file to read.
    :type path: Path
    :returns: One source string per Markdown cell, in cell order.
    :rtype: list[str]

    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]


def _code_sources(notebook: dict) -> list[str]:
    """Collect the source of every code cell of one parsed notebook.

    :param notebook: A notebook, as parsed from its ``.ipynb`` JSON.
    :type notebook: dict
    :returns: One source string per code cell, in cell order.
    :rtype: list[str]

    """
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _lacks_inline_magic(notebook: dict) -> bool:
    """Tell whether a notebook draws figures without ``%matplotlib inline`` first.

    :param notebook: A notebook, as parsed from its ``.ipynb`` JSON.
    :type notebook: dict
    :returns: ``True`` if some code cell draws figures and the first non-comment,
        non-blank line of the first code cell is not ``%matplotlib inline``.
    :rtype: bool

    """
    sources = _code_sources(notebook=notebook)
    if not any(_PLOTTING_PATTERN.search(source) for source in sources):
        return False
    first_statement = next(
        (
            line.strip()
            for line in sources[0].splitlines()
            if line.strip() and not line.strip().startswith(_COMMENT_PREFIX)
        ),
        None,
    )
    return first_statement != _INLINE_MAGIC


def _classified_lines(source: str) -> list[tuple[str, str]]:
    """Label every line of one Markdown cell by its structural role.

    :param source: A Markdown cell's full source.
    :type source: str
    :returns: One ``(kind, line)`` pair per line, where ``kind`` is ``"blank"``,
        ``"code"`` (inside or delimiting a fence), ``"math"`` (inside or delimiting a
        display-math block), ``"heading"``, ``"list"``, ``"table"``, or ``"prose"``.
    :rtype: list[tuple[str, str]]

    """
    labeled = []
    in_fence = in_math = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_fence or _FENCE_PATTERN.match(line):
            # A fence line both opens and closes a code block.
            if _FENCE_PATTERN.match(line):
                in_fence = not in_fence
            labeled.append(("code", line))
        elif in_math or stripped.startswith(_DISPLAY_MATH_DELIMITER):
            # "$$" alone opens or closes a block; "$$...$$" is a one-line block.
            one_line_block = (
                len(stripped) > len(_DISPLAY_MATH_DELIMITER)
                and stripped.startswith(_DISPLAY_MATH_DELIMITER)
                and stripped.endswith(_DISPLAY_MATH_DELIMITER)
            )
            if not one_line_block and _DISPLAY_MATH_DELIMITER in stripped:
                in_math = not in_math
            labeled.append(("math", line))
        elif not stripped:
            labeled.append(("blank", line))
        elif _HEADING_PATTERN.match(line):
            labeled.append(("heading", line))
        elif _LIST_ITEM_PATTERN.match(line):
            labeled.append(("list", line))
        elif stripped.startswith(_TABLE_ROW_PREFIX):
            labeled.append(("table", line))
        else:
            labeled.append(("prose", line))
    return labeled


def _wrapped_lines(source: str) -> list[str]:
    """Find every prose line that continues the line before it.

    :param source: A Markdown cell's full source.
    :type source: str
    :returns: Each offending continuation line, in order.
    :rtype: list[str]

    """
    labeled = _classified_lines(source=source)
    return [
        line
        for (previous_kind, _), (kind, line) in zip(labeled, labeled[1:])
        if kind == "prose" and previous_kind in ("prose", "list")
    ]


def _prose_outside_fences(source: str) -> str:
    """Drop every fenced code block from one Markdown cell.

    :param source: A Markdown cell's full source.
    :type source: str
    :returns: The cell's lines outside fenced code, newline-joined.
    :rtype: str

    """
    return "\n".join(
        line for kind, line in _classified_lines(source=source) if kind != "code"
    )


# ========================================================================================
# Tests
# ========================================================================================


def test_notebooks_are_found() -> None:
    """The notebook glob finds the tutorials, so the lints below never run empty."""
    assert NOTEBOOK_PATHS, "no notebooks found under notebooks/"


@pytest.mark.parametrize(
    argnames="path", argvalues=NOTEBOOK_PATHS, ids=lambda p: p.name
)
def test_markdown_is_not_hard_wrapped(path: Path) -> None:
    """Each Markdown paragraph and list item is a single source line."""
    wrapped = [
        line
        for source in _markdown_sources(path=path)
        for line in _wrapped_lines(source=source)
    ]
    assert not wrapped, f"{path.name} has hard-wrapped Markdown lines: {wrapped}"


@pytest.mark.parametrize(
    argnames="path", argvalues=NOTEBOOK_PATHS, ids=lambda p: p.name
)
def test_math_is_not_typed_as_code(path: Path) -> None:
    """No backticked span in Markdown contains ``^`` or ``_{``."""
    spans = [
        match.group(1)
        for source in _markdown_sources(path=path)
        for match in _CODE_SPAN_PATTERN.finditer(_prose_outside_fences(source=source))
        if any(marker in match.group(1) for marker in _MATH_IN_CODE_MARKERS)
    ]
    assert not spans, f"{path.name} types math as code; render it with $...$: {spans}"


@pytest.mark.parametrize(
    argnames=("source", "expected"),
    argvalues=[
        ("One paragraph on one line.", []),
        ("A paragraph wrapped\nby hand.", ["by hand."]),
        ("- a list item\n  continued by hand", ["  continued by hand"]),
        ("- one item\n- another item", []),
        ("## Heading\nA paragraph right below it.", []),
        ("A paragraph.\n\n$$\nx = 1\ny = 2\n$$\n\nAnother.", []),
        ("A paragraph.\n\n$$x^2$$\n\nAnother.", []),
        ("```bash\npip install pypft\npython -m pip install pypft\n```", []),
        ("| a | b |\n| - | - |\n| 1 | 2 |", []),
    ],
    ids=[
        "single-line",
        "wrapped-paragraph",
        "wrapped-list-item",
        "list",
        "heading",
        "display-math-block",
        "one-line-display-math",
        "fence",
        "table",
    ],
)
def test_wrap_detection(source: str, expected: list[str]) -> None:
    """The wrap check flags only prose continuation lines."""
    assert _wrapped_lines(source=source) == expected


@pytest.mark.parametrize(
    argnames="path", argvalues=NOTEBOOK_PATHS, ids=lambda p: p.name
)
def test_figures_are_drawn_inline(path: Path) -> None:
    """A notebook that draws figures opens with ``%matplotlib inline``."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert not _lacks_inline_magic(
        notebook=notebook
    ), f"{path.name} draws figures; start its first code cell with {_INLINE_MAGIC}"


def _notebook_of(*sources: str) -> dict:
    """Build a minimal parsed notebook whose code cells hold the given sources.

    :param sources: The source of each code cell, in order.
    :type sources: str
    :returns: A notebook dictionary with one code cell per source.
    :rtype: dict

    """
    return {
        "cells": [
            {"cell_type": "code", "source": source.splitlines(keepends=True)}
            for source in sources
        ]
    }


@pytest.mark.parametrize(
    argnames=("sources", "expected"),
    argvalues=[
        (("import matplotlib.pyplot as plt", "plt.show()"), True),
        (("import numpy as np", "pypft.plot_signal(signal=s)"), True),
        (("import numpy as np", "signal.plot(ax=ax)"), True),
        (
            ("x = 1", "pypft.forward_pft_traced(f=v, grid=g, visualize_steps=True)"),
            True,
        ),
        (("%matplotlib inline\nimport matplotlib.pyplot as plt",), False),
        (("# Draw figures inline\n%matplotlib inline\n\nimport matplotlib",), False),
        (("import matplotlib\n%matplotlib inline",), True),
        (("import numpy as np", "print(np.pi)"), False),
    ],
    ids=[
        "matplotlib-without-magic",
        "plot-signal-without-magic",
        "signal-plot-without-magic",
        "traced-figures-without-magic",
        "magic-first",
        "comment-then-magic",
        "magic-after-import",
        "no-figures",
    ],
)
def test_inline_magic_detection(sources: tuple[str, ...], expected: bool) -> None:
    """The inline check flags only figure-drawing notebooks missing the magic."""
    assert _lacks_inline_magic(notebook=_notebook_of(*sources)) is expected


def test_math_in_code_detection_ignores_fences() -> None:
    """Backticked math is flagged in prose but not inside a fenced code block."""
    source = "Use `x^2` here.\n\n```python\ny = `z_{1}`\n```"
    spans = _CODE_SPAN_PATTERN.findall(_prose_outside_fences(source=source))
    assert spans == ["x^2"]
