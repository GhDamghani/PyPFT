"""Every tutorial notebook is committed stripped: no outputs, no metadata."""

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
    """A notebook carries no notebook/cell metadata, outputs, or execution counts."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["metadata"] == {}
    for cell in notebook["cells"]:
        assert cell.get("metadata", {}) == {}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
