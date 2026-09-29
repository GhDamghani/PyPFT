"""Reduce Jupyter notebooks to their simplest, most generic on-disk form.

Opening and saving a notebook in Jupyter or VS Code stamps it with the machine's
interpreter (``metadata.language_info.version``, ``nbconvert_exporter``), the kernel
that ran it, cell execution counts, freshly generated cell ids, and every rendered
output. None of that belongs in a version-controlled notebook: it identifies the
machine the notebook was authored on, and it produces diff noise on every save.

This module strips all of it, leaving only the notebook's cells and their source:

.. code-block:: json

    {
     "cells": [ ... ],
     "metadata": {},
     "nbformat": 4,
     "nbformat_minor": 5
    }

Cell ids are the one field renumbered rather than removed: nbformat 4.5 requires an
``id`` on every cell, and a notebook without them raises ``MissingIDFieldWarning``
when read -- which ``nbmake`` turns into a test failure under this project's
``filterwarnings = ["error"]``. Each cell's id is set to its own index (``"0"``,
``"1"``, ...), so ids stay valid and unique without carrying an editor's random
identifiers into the diff.

It deliberately depends on nothing outside the standard library -- not ``nbformat``,
not ``pypft`` -- so it can run as a pre-commit hook under a bare interpreter, even
when the project environment is missing or out of date.

Usage:
    python scripts/strip_notebook.py [--check] [<notebook or directory> ...]

With no paths, every notebook under ``notebooks/`` is stripped in place.
``--check`` writes nothing, lists every notebook that would change, and exits 1 if
there is any.
"""

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

# ======================================================================================
# Constants
# ======================================================================================

#: Filename suffix identifying a Jupyter notebook.
NOTEBOOK_SUFFIX = ".ipynb"

#: Paths searched when the command line names none: the repository's tutorial
#: notebooks. Every other notebook in the tree is generated or gitignored.
DEFAULT_SEARCH_PATHS = (Path("notebooks"),)

#: Directory names never descended into while searching for notebooks: git's own
#: metadata and Jupyter's autosave directory.
EXCLUDED_DIRECTORY_NAMES = frozenset({".git", ".ipynb_checkpoints"})

#: Indentation width Jupyter itself writes notebooks with.
JSON_INDENT = 1

#: Keys kept on a non-code cell, in the order Jupyter writes them.
_CANONICAL_CELL_KEYS = ("cell_type", "id", "metadata", "source")

#: Keys kept on a code cell, in the order Jupyter writes them.
_CANONICAL_CODE_CELL_KEYS = (
    "cell_type",
    "execution_count",
    "id",
    "metadata",
    "outputs",
    "source",
)


# ======================================================================================
# Stripping
# ======================================================================================


def strip_notebook_dict(notebook: dict) -> dict:
    """Rebuild a parsed notebook in its simplest, most generic form.

    Drop the whole top-level ``metadata`` block (kernelspec and language info alike),
    every cell's ``metadata`` and ``attachments``, and every code cell's ``outputs``
    and ``execution_count``, and renumber every cell's ``id`` to its own index. Cell
    source and ordering are preserved verbatim, as are ``nbformat`` and
    ``nbformat_minor``.

    :param notebook: The parsed contents of a ``.ipynb`` file.
    :type notebook: dict
    :return: A new notebook dictionary in canonical form.
    :rtype: dict
    :raises ValueError: If ``notebook`` is not a notebook: not a mapping, without a
        list of cells, or with a cell that is not a mapping.
    """
    if not isinstance(notebook, dict):
        raise ValueError(
            f"Notebook must be a JSON object, got {type(notebook).__name__}."
        )
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook must have a 'cells' list.")

    stripped_cells: list[dict] = []
    # Rebuild every cell from scratch, so unknown keys are dropped rather than kept.
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(
                f"Cell {cell_index} must be a JSON object, got {type(cell).__name__}."
            )
        stripped_cells.append(_strip_cell(cell=cell, cell_index=cell_index))

    return {
        "cells": stripped_cells,
        "metadata": {},
        "nbformat": notebook.get("nbformat", 4),
        "nbformat_minor": notebook.get("nbformat_minor", 5),
    }


def strip_notebook_text(notebook_text: str) -> str:
    """Strip a notebook's raw file contents, preserving its line endings.

    :param notebook_text: The full text of a ``.ipynb`` file, read without newline
        translation.
    :type notebook_text: str
    :return: The canonical text the file should hold, ending in a single newline.
    :rtype: str
    :raises ValueError: If the text is not valid JSON, or not a notebook.
    """
    try:
        notebook = json.loads(s=notebook_text)
    except json.JSONDecodeError as decode_error:
        raise ValueError(f"Not valid JSON: {decode_error}") from decode_error

    stripped = strip_notebook_dict(notebook=notebook)
    # Match the file's own line endings, so a CRLF checkout keeps its CRLFs.
    line_ending = "\r\n" if "\r\n" in notebook_text else "\n"
    stripped_text = (
        json.dumps(obj=stripped, indent=JSON_INDENT, ensure_ascii=False) + "\n"
    )
    return stripped_text.replace("\n", line_ending)


def strip_notebook_file(file_path: Path, write: bool = True) -> bool:
    """Rewrite a notebook file in canonical form, leaving it untouched if it is clean.

    A file that is already canonical is not written at all, which keeps the operation
    idempotent and its modification time stable.

    :param file_path: The ``.ipynb`` file to strip, rewritten in place.
    :type file_path: Path
    :param write: Whether to write the stripped result. ``False`` only reports whether
        the file would change, which is what ``--check`` needs.
    :type write: bool
    :return: Whether the file's contents differ from their canonical form.
    :rtype: bool
    :raises ValueError: If the file is not valid JSON, or not a notebook.
    :raises OSError: If the file cannot be read or written.
    """
    with open(file=file_path, mode="r", encoding="utf-8", newline="") as notebook_file:
        original_text = notebook_file.read()

    try:
        stripped_text = strip_notebook_text(notebook_text=original_text)
    except ValueError as value_error:
        raise ValueError(f"{file_path}: {value_error}") from value_error

    if stripped_text == original_text:
        return False
    if write:
        with open(
            file=file_path, mode="w", encoding="utf-8", newline=""
        ) as notebook_file:
            notebook_file.write(stripped_text)
    return True


def iter_notebook_paths(paths: Sequence[Path]) -> Iterator[Path]:
    """Expand a mix of notebook files and directories into notebook file paths.

    A directory is searched recursively, skipping ``EXCLUDED_DIRECTORY_NAMES``; a file
    is yielded as given, whether or not it lies under an excluded directory, since
    naming it explicitly is an explicit request. A directory's results are sorted, so a
    run's output is reproducible.

    :param paths: The files and/or directories named on the command line.
    :type paths: Sequence[Path]
    :return: An iterator over the ``.ipynb`` files to strip.
    :rtype: Iterator[Path]
    """
    # Walk every requested path, expanding directories and passing files through.
    for path in paths:
        if path.is_dir():
            for notebook_path in sorted(path.rglob(pattern=f"*{NOTEBOOK_SUFFIX}")):
                if not _is_excluded(notebook_path=notebook_path):
                    yield notebook_path
        else:
            yield path


def _strip_cell(cell: dict, cell_index: int) -> dict:
    """Rebuild one cell with only its canonical keys, in Jupyter's key order.

    :param cell: A single parsed notebook cell.
    :type cell: dict
    :param cell_index: The cell's position in the notebook, used as its ``id``.
    :type cell_index: int
    :return: A new cell dictionary holding only canonical keys.
    :rtype: dict
    """
    cell_type = cell.get("cell_type", "code")
    stripped: dict = {
        "cell_type": cell_type,
        "id": str(cell_index),
        "metadata": {},
        "source": cell.get("source", []),
    }
    if cell_type == "code":
        stripped["execution_count"] = None
        stripped["outputs"] = []
        key_order = _CANONICAL_CODE_CELL_KEYS
    else:
        key_order = _CANONICAL_CELL_KEYS
    return {key: stripped[key] for key in key_order}


def _is_excluded(notebook_path: Path) -> bool:
    """Report whether a discovered notebook lies under an excluded directory.

    :param notebook_path: A notebook path found while searching a directory.
    :type notebook_path: Path
    :return: Whether any directory in the path is excluded.
    :rtype: bool
    """
    return any(part in EXCLUDED_DIRECTORY_NAMES for part in notebook_path.parts[:-1])


# ======================================================================================
# Command-line interface
# ======================================================================================


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the strip_notebook CLI.

    :param argv: The argument strings to parse, or ``None`` to parse ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    """
    default_paths = ", ".join(str(path) for path in DEFAULT_SEARCH_PATHS)
    parser = argparse.ArgumentParser(
        description=(
            "Strip Jupyter notebooks down to cells and source: no kernelspec, no "
            "interpreter version, no outputs, no execution counts, no cell metadata, "
            "and cell ids renumbered to each cell's index."
        )
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help=(
            "Notebook files and/or directories to search recursively. Defaults to "
            f"{default_paths}."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report which notebooks are not in canonical form and exit 1 if any are, "
            "without writing anything."
        ),
    )
    return parser.parse_args(args=argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strip_notebook command-line interface.

    :param argv: The argument strings to parse, or ``None`` to parse ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: ``1`` if ``--check`` found a notebook needing stripping, else ``0``.
    :rtype: int
    :raises ValueError: If a notebook is not valid JSON, or not a notebook.
    :raises OSError: If a notebook cannot be read or written.
    """
    args = _parse_args(argv=argv)
    paths = args.paths if args.paths else list(DEFAULT_SEARCH_PATHS)

    changed_paths = [
        notebook_path
        for notebook_path in iter_notebook_paths(paths=paths)
        if strip_notebook_file(file_path=notebook_path, write=not args.check)
    ]

    verb = "would be stripped" if args.check else "stripped"
    # Name every affected notebook, so a failed hook run says what it touched.
    for changed_path in changed_paths:
        print(f"{verb}: {changed_path}")
    if not changed_paths:
        print("All notebooks are already in canonical form.")
    return 1 if args.check and changed_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
