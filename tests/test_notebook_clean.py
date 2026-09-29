"""Every tutorial notebook is committed stripped: no outputs, no metadata.

This is the canonical form ``scripts/strip_notebook.py`` (the pre-commit hook) writes,
including its cell ids, each renumbered to the cell's own index.
"""

import json
from pathlib import Path

import pytest

# ========================================================================================
# Constants
# ========================================================================================

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOK_PATHS = sorted(NOTEBOOKS_DIR.glob(pattern="*.ipynb"))


# ========================================================================================
# Tests
# ========================================================================================


@pytest.mark.parametrize(
    argnames="path", argvalues=NOTEBOOK_PATHS, ids=lambda p: p.name
)
def test_notebook_is_stripped(path: Path) -> None:
    """A notebook carries no metadata, outputs, or execution counts; ids are indices."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"] == {}
    for index, cell in enumerate(notebook["cells"]):
        assert cell["id"] == str(index)
        assert cell.get("metadata", {}) == {}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
