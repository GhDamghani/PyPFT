# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT ("Polar Fourier transform toolkit with a modular PyPFT facade") was substantially rewritten from
scratch in the `start over` commit (deccca8). The previous implementation had a full package layout
(`backends/`, `cli/`, `core/`, `dft/`, `dht/`, `fields/`, `grids/`, `idft/`, `io/`, docs, benchmarks,
notebooks, release scripts) — all of that was deleted. The current tree contains only an empty
`src/pypft/__init__.py` stub and a test fixture image (`tests/samples/lena.tif`); there is no test suite,
CLI, or documentation yet. Do not assume any prior architecture, module, or API still exists — check the
current file tree before referencing paths from git history.

## Environment and commands

This project uses **uv** for dependency management (`uv.lock` is committed) and targets **Python 3.14
exclusively** (`requires-python = ">=3.14,<3.15"` in `pyproject.toml`).

- Install/sync dependencies: `uv sync` (add `--group dev` deps are included by default via dependency-groups)
- Run the test suite: `uv run pytest` (tests live under `tests/`, per `.vscode/settings.json`)
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Build the package: `uv build` (setuptools backend)

There is no lint/format/type-check tool configured in `pyproject.toml` yet (no `[tool.ruff]`,
`[tool.black]`, or `[tool.mypy]` section), and `.github/` currently has no CI workflows.

## Dependencies

Runtime: `numpy`, `scipy`, `numba`, `matplotlib`. Dev group: `pytest`, `pytest-cov`, `sphinx` + `furo`
(for docs, though no `docs/` source tree exists yet).

## Conventions

- Editor ruler / preferred line length is 80 columns (`.vscode/settings.json`).
- License is BSD-3-Clause.

## Local, gitignored data

`.local_files/` (gitignored) holds local example/validation data (`example_files/`, `example_roundtrip/`,
`sources/`, `validation/`, a `grids` file, and a `.local_files.7z` archive). This is scratch/reference
material for local development, not part of the package or test suite.
