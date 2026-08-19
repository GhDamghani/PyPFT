# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying paper.

The package was substantially rewritten from scratch in the `start over` commit (deccca8), which deleted a
full prior layout (`backends/`, `cli/`, `core/`, `dft/`, `dht/`, `fields/`, `grids/`, `idft/`, `io/`, docs,
benchmarks, notebooks, release scripts). It is being rebuilt incrementally; currently only
`src/pypft/utils/validators.py` exists as real code. There is no test suite, CLI, DHT/DFT implementation,
or documentation yet. Do not assume any prior architecture, module, or API still exists — check the current
file tree before referencing paths from git history.

## Environment and commands

Dependency management is via **uv** (`uv.lock` is committed); the project targets **Python 3.14
exclusively** (`requires-python = ">=3.14,<3.15"`).

- Install/sync the environment: `uv sync` — run this after every pull/branch switch, since `pyproject.toml`
  changes frequently right now.
- Run the test suite: `uv run pytest` (tests live under `tests/`)
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Build the package: `uv build`
- Add/upgrade a dependency (don't hand-edit version pins): `uv add "<pkg>>=X.Y.Z"` or
  `uv add --group dev "<pkg>>=X.Y.Z"`, then `uv lock` / `uv sync`.
- Format/lint/type-check (all configured in `pyproject.toml`, no wrapper script exists):
  `uv run black src`, `uv run isort src`, `uv run flake8 src`, `uv run pyright`, `uv run vulture src`

`pyproject.toml`'s `[project]` table must keep `dependencies = [...]` before `[project.urls]` — TOML
otherwise attaches a bare `dependencies` key to whichever table header precedes it (this previously broke
the build: `numba`/`scipy` silently vanished from the resolved lock and `uv sync` failed with a setuptools
`project.urls.dependencies` validation error).

## Architecture: the validators module

`src/pypft/utils/validators.py` is the one substantial file in the codebase today, and its module docstring
is the authoritative spec for how validation is done project-wide — read it before adding a new validator.
Key points:

- One class per type, named `<Type>Validator` (e.g. `IntValidator`, `PathValidator`), holding
  `@staticmethod`s. Validators for locally-defined (in-package) types live on the class where that type is
  defined, to avoid circular imports — not in this shared module.
- Methods are named `type_is_<typename>` (type-validators, raise `TypeError`) or
  `value_<is|has|should|...>_<condition>` (value-validators, raise `ValueError`, or an `OSError` subclass
  for filesystem-state checks like "path writable").
- Validators only validate — they never mutate/replace their arguments — and assume prior type-validation
  has already run (no redundant `isinstance` re-checks inside a value-validator).
- Classes/methods are ordered to mirror PEP 8 import order: builtins → stdlib → third-party, with
  type-validators before value-validators, simple types before composite types, single-input before
  multi-input validators.
- VS Code snippets in `.vscode/helpers.code-snippets` scaffold new validators: `v-type` for a
  type-validator, `v-value` for a value-validator.

## Conventions (from `README.md`)

- **Errors/warnings**: raise exceptions for invalid state — prefer purpose-built subclasses of builtin
  exceptions (e.g. `Val1ValueError(ValueError)`) over generic ones when callers need to distinguish cases;
  use the `warnings` module (not silent fallback) when degrading gracefully instead of raising.
- **Type annotations**: required on all function signatures, with input validation via the
  `*Validator` classes above.
- **Docstrings**: reStructuredText (PEP 287 / Sphinx `:param:`/`:type:`/`:raises:` style), not
  Google/NumPy style — see any method in `validators.py` for the pattern.
- **Line length**: `pyproject.toml`'s `[tool.black]`/`[tool.flake8]` enforce 88 chars — note `README.md`
  and `.vscode/settings.json` (editor rulers, `rewrap.wrappingColumn`) both say 90; treat the actually
  configured values in `pyproject.toml` as authoritative for tooling.
- **Code-sectioning comments**: `# ` + repeated character to fill the line width — `=` for a top-level
  section, `-` for a subsection, `*`/`.` for the same one indent level in. Snippets: `@ section`,
  `@ subsection`, `@ isection`, `@ isubsection` in `.vscode/helpers.code-snippets`.

## Local, gitignored data

`.local_files/` (gitignored) holds local example/validation data (`example_files/`, `example_roundtrip/`,
`sources/`, `validation/`, a `grids` file, and a `.local_files.7z` archive) — scratch/reference material for
local development, not part of the package or test suite.
