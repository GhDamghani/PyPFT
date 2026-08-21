"""Machine-checks the notebook citation discipline (develop_plan.md §2.4).

Notebook Markdown cells write a citation's inline label out literally (e.g.
"...is self-inverse [Baddour2019a, Eq. 41]."), since Markdown cells cannot
execute code to interpolate it. Two things are checked here instead: every
such label actually resolves to a ``pypft.references.Reference`` member, and
every notebook that cites anything ends with a ``bibliography(...)`` call
that lists exactly the members it cited -- no orphans, no omissions.
"""

import json
import re
from pathlib import Path

import pytest

from pypft.references import Reference

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
_LABEL_PATTERN = re.compile(r"\[([A-Za-z]+\d{4}[a-z]?)(?:,[^\]]*)?\]")
_KEYS_TO_MEMBERS = {member.value.key: member for member in Reference}


def _notebook_paths() -> list[Path]:
    """List every tracked notebook, in a stable order.

    :returns: The tutorial notebooks' paths, sorted by name.
    :rtype: list[Path]

    """
    return sorted(_NOTEBOOKS_DIR.glob("*.ipynb"))


def _cell_sources(notebook: dict, cell_type: str) -> list[str]:
    """Concatenate every source line of a notebook's cells of one type.

    :param notebook: A parsed ``.ipynb`` document.
    :type notebook: dict
    :param cell_type: ``"markdown"`` or ``"code"``.
    :type cell_type: str
    :returns: One joined-source string per matching cell, in cell order.
    :rtype: list[str]

    """
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == cell_type
    ]


def _cited_keys(notebook: dict) -> set[str]:
    """Collect every bracketed citation label cited in a notebook's Markdown.

    :param notebook: A parsed ``.ipynb`` document.
    :type notebook: dict
    :returns: The set of citation keys (without brackets) cited.
    :rtype: set[str]

    """
    keys = set()
    for text in _cell_sources(notebook, "markdown"):
        keys.update(match.group(1) for match in _LABEL_PATTERN.finditer(text))
    return keys


@pytest.mark.parametrize("path", _notebook_paths(), ids=lambda p: p.name)
def test_every_cited_label_resolves_to_a_reference(path):
    """No notebook cites a key that isn't a real ``Reference`` member."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    unknown = _cited_keys(notebook) - set(_KEYS_TO_MEMBERS)
    assert not unknown, f"{path.name} cites unknown reference(s): {unknown}"


@pytest.mark.parametrize("path", _notebook_paths(), ids=lambda p: p.name)
def test_citing_notebooks_end_with_a_matching_bibliography_call(path):
    """A notebook that cites anything ends with the matching bibliography."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cited = _cited_keys(notebook)
    if not cited:
        return

    code_cells = _cell_sources(notebook, "code")
    assert code_cells, f"{path.name} cites references but has no code cells"
    final_cell = code_cells[-1]
    assert "bibliography(" in final_cell, (
        f"{path.name} cites references but its final cell doesn't render "
        "the bibliography"
    )

    rendered = {member.value.key for member in Reference if member.name in final_cell}
    assert rendered == cited, (
        f"{path.name}: bibliography cell renders {rendered}, but the "
        f"notebook cites {cited}"
    )
