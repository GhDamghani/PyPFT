# Contributing to PyPFT

Thank you for considering a contribution to PyPFT. This guide covers how to report issues, how to set up a development environment, how to run the quality gate, the project's conventions, and how to open a pull request. A step-by-step setup tutorial also lives in the documentation (`docs/contributing.rst`).

## Reporting issues

Open an issue on [GitHub](https://github.com/GhDamghani/PyPFT/issues). A useful report includes:

- what you ran (a minimal, self-contained code snippet),
- what you expected, and what happened instead (the full traceback, if any),
- your operating system, Python version, and PyPFT version (`pip show pypft`, or `uv version` in a development checkout).

For numerical problems, include the grid parameters (`n_radial`, `n_angular`, `R`) and, if possible, the error you measured against a known analytic answer.

## Setting up a development environment

PyPFT's dependencies are managed with [`uv`](https://docs.astral.sh/uv/), and `uv.lock` is committed.

1. Install `uv`.
2. Fork the repository on GitHub, then clone your fork:

   ```bash
   git clone https://github.com/<your-username>/PyPFT.git
   cd PyPFT
   ```

3. From the repository root, create the environment (runtime and development dependencies):

   ```bash
   uv sync
   ```

4. Install the pre-commit hook, which strips every notebook under `notebooks/` to its bare form (no outputs, no execution counts, no metadata, cell ids renumbered to each cell's index) on each commit:

   ```bash
   uv run pre-commit install
   ```

   When the hook changes a notebook, the commit stops so you can review and re-stage it. The hook runs `scripts/strip_notebook.py`, which uses only the standard library, so it works under a bare `python` even when the project environment is missing or out of date. To strip every notebook by hand, or only check them (`--check` exits 1 if any notebook would change):

   ```bash
   python scripts/strip_notebook.py --check
   ```

Run `uv sync` again after every pull or branch switch, so your environment always matches `pyproject.toml`.

### Supported Python versions and dependency floors

PyPFT supports every CPython listed in `supported_python_versions.txt` (currently 3.12, 3.13, and 3.14). Development happens on the oldest of them: `.python-version` pins 3.12, so `uv sync` creates a 3.12 environment, installing the interpreter if needed.

That environment also runs against the *oldest* supported release of each runtime dependency. `pyproject.toml` publishes a per-interpreter floor for each dependency (the oldest release shipping a wheel for that CPython) with no upper bound, and its `[tool.uv] constraint-dependencies` table pins your environment to exactly those floors, so the lower bounds PyPFT claims are the ones you test against. CI runs the full quality gate once, on that development setup (`.github/workflows/ci.yml`), and runs the tests and notebooks on every supported interpreter twice: once at those floors ("minimum") and once after upgrading to the newest releases ("latest") (`.github/workflows/tests.yml`). Both run on Linux.

To reproduce the "latest" leg locally, upgrade the runtime dependencies past the floors and run the gate without re-syncing (`--no-config` makes `uv pip` ignore the constraint table):

```bash
uv pip install --no-config --upgrade matplotlib numba llvmlite numpy opencv-python-headless scipy
./scripts/Invoke-QualityGate.ps1 -NoSync
```

Run `uv sync` afterwards to return to the floors. To test on a different interpreter, set `UV_PYTHON` (for example `UV_PYTHON=3.14`) before `uv sync`.

To add or upgrade a dependency, don't hand-edit the version pins; use `uv add "<pkg>>=X.Y.Z"` (runtime) or `uv add --group dev "<pkg>>=X.Y.Z"` (development), then `uv lock` and `uv sync`. A new runtime dependency also needs a per-interpreter wheel floor and a matching `constraint-dependencies` entry, following the comments in `pyproject.toml`.

### Development dependencies

- `black`, `isort`: code formatting and import sorting.
- `flake8` with `flake8-pyproject`, `flake8-docstrings` (missing-docstring checks), and `flake8-rst-docstrings` (reStructuredText docstring validation).
- `pyright`: static type checking.
- `vulture`, `pytest-vulture`: dead-code detection.
- `pytest`, `pytest-benchmark`, `pytest-cov`: tests, benchmarks, and coverage.
- `pre-commit`: runs the notebook-stripping commit hook (`.pre-commit-config.yaml`, which calls `scripts/strip_notebook.py`).
- `nbmake`: executes and checks every tutorial notebook as part of the test suite.
- `sphinx`, `furo`, `myst-nb`: the documentation build.
- `notebook`, `ipython`, `ipdb`: interactive work, notebook authoring, and debugging.
- `pyment`: generates reStructuredText docstring templates from signatures.
- `setuptools`, `wheel`: packaging.
- `tqdm`: progress bars in development scripts.
- `pillow`: reads and writes the image files of the test fixtures (`scripts/make_maze.py`).

### Recommended editor setup

The project does not ship editor configuration. If you use VS Code, these extensions match the project's tooling: Python (`ms-python.python`), Pylance (`ms-python.vscode-pylance`), Python Debugger (`ms-python.debugpy`), Black Formatter (`ms-python.black-formatter`), Flake8 (`ms-python.flake8`), isort (`ms-python.isort`), Python Indent (`KevinRose.vsc-python-indent`), autoDocstring (`njpwerner.autodocstring`, set to the reStructuredText/Sphinx format), Code Spell Checker (`streetsidesoftware.code-spell-checker`, American English), Rewrap Revived (`dnut.rewrap-revived`, for wrapping comments and docstrings), Markdown Table Prettifier (`darkriszty.markdown-table-prettify`), and GitHub Markdown Preview (`bierner.github-markdown-preview`). Set your editor's ruler and wrapping column to 88.

## Running tests and the quality gate

- Run the whole test suite (this also executes every notebook under `notebooks/` via `nbmake`):

  ```bash
  uv run pytest
  ```

- Run a single test:

  ```bash
  uv run pytest tests/path/to/test_file.py::test_name
  ```

- Run only the notebooks:

  ```bash
  uv run pytest --nbmake notebooks/
  ```

- Run the full quality gate — the same script CI runs (pytest, the notebook suite, `black --check`, `isort --check-only`, `flake8`, `pyright`, `vulture`, the docs build, and `uv build`, stopping at the first failure):

  ```powershell
  ./scripts/Invoke-QualityGate.ps1
  ```

- Build the documentation (warnings fail the build):

  ```bash
  uv run sphinx-build -W docs docs/_build
  ```

- Benchmark the implementation strategies (not part of the test suite; reports are written to `.local_files/benchmarks/results/`):

  ```bash
  uv run python benchmarks/run_dht_benchmarks.py
  ```

  `benchmarks/run_dft_benchmarks.py` and `benchmarks/run_pft_benchmarks.py` work the same way.

- Regenerate the maze test fixture (`tests/samples/maze_polar.tif`, with `maze_uniform_polar.tif`, `maze_cartesian.png` and `maze_source.png` next to it). `scripts/make_maze.py` carves a perfect circular maze (concentric rings of cells around a courtyard) from a fixed seed and evaluates its walls analytically twice: at a `PolarGrid`'s own sample points (`maze_polar.tif`, which feeds `forward_pft` directly) and on a uniform polar grid with the same spokes and sample counts (`maze_uniform_polar.tif`, which goes through `resample_uniform_polar` first). A third helper, `sample_maze_rings`, computes the maze's harmonics on each harmonic's own true rings, the input of the ring-consistent route (`forward_pft_ring`); it writes no file, so the route's input is generated on the fly by the notebook and tests. The script writes all four files; its defaults reproduce the committed files exactly, which `tests/test_make_maze.py` checks. The default grid triggers `check_adequacy`'s warning, which is expected: the maze is a display and regression image, not an accuracy reference.

  ```bash
  uv run python scripts/make_maze.py
  ```

  Run `uv run python scripts/make_maze.py --help` for the options (maze size `--rings` and `--inner-sectors`, `--seed`, `--wall-thickness`, the grid's `--n-radial`/`--n-angular`/`--radius`, `--maze-fraction`, `--image-size`, and `--output-dir`).

Tests live under `tests/`, mirroring the package layout of `src/pypft/` (e.g. `tests/dht/` for `src/pypft/dht/`). Warnings are errors in the test suite, so any `warnings.warn` in `src/` needs a matching `pytest.warns` test.

## Conventions

The code style is [PEP 8](https://peps.python.org/pep-0008/), with one exception: the maximum line length is 88 rather than 79, as recommended by [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length). Try to keep comments, docstrings, and strings within 88 characters as well.

### Constant extraction

Extract constants as much as possible. A constant used only within one module goes at the top of that module; a constant shared more broadly goes at whatever level actually shares it.

### Errors and warnings

Invalid state raises an exception. When callers need to tell failure cases apart, define purpose-built subclasses of the built-in exceptions ([inheriting from built-in exceptions](https://docs.python.org/3/library/exceptions.html#inheriting-from-built-in-exceptions)) instead of raising the generic one. When degrading gracefully instead of raising, emit a warning with the `warnings` module rather than falling back silently.

<details>
<summary>Example</summary>

```python
import warnings


class Val1ValueError(ValueError):
    """Raised when val1 fails validation."""


class Val2ValueError(ValueError):
    """Raised when val2 fails validation."""


def validate(val1: int, val2: int) -> None:
    if val1 < 0:
        raise Val1ValueError(f"val1 must be non-negative, got {val1!r}")
    if val2 == 0:
        raise Val2ValueError(f"val2 must not be zero, got {val2!r}")
    if val1 > 1000:
        warnings.warn(f"val1={val1} is unusually large", stacklevel=2)


def process(val1: int, val2: int) -> float:
    DEFAULT_VAL1 = 5
    try:
        validate(val1, val2)
    except Val1ValueError:
        warnings.warn(
            f"val1={val1} is invalid, substituting {DEFAULT_VAL1}", stacklevel=2
        )
        val1 = DEFAULT_VAL1
    return val1 / val2
```

</details>

### Keyword arguments

Wherever a call site can pass an argument by keyword, it should: prefer `PolarGrid(n_radial=383, n_angular=15, R=40.0)` over `PolarGrid(383, 15, 40.0)`. This applies to calls into PyPFT's own API and to third-party calls (NumPy, SciPy, and so on). The only exemptions are callees whose own signature forces positional-only arguments.

### Type annotations and validators

Every function signature carries type annotations, and every public input is validated. Validators for shared and third-party types live in `src/pypft/utils/validators.py`; read its module docstring before adding one. Validators for types defined inside PyPFT live on the class where that type is defined (for example, `PolarGrid`'s own type-validator lives in `src/pypft/grid.py`), to avoid circular imports. Type-validators are named `type_is_<typename>` and raise `TypeError`; value-validators are named `value_<is|has|should|...>_<condition>` and raise `ValueError`.

### Docstrings

Docstrings follow [PEP 287](https://peps.python.org/pep-0287/) and are written in [reStructuredText](https://docutils.sourceforge.io/rst.html), using Sphinx's `:param:`/`:type:`/`:returns:`/`:raises:` fields. Don't use Sphinx cross-reference roles such as `:class:` or `:func:`; write names as double-backtick literals instead. Docstrings, comments, and notebooks describe the current design and its rationale only, not its history; a longer technical rationale belongs in `DESIGN_NOTES.md`, referenced by section from the code it explains.

### Notebook citations

A tutorial notebook that states a result from a paper cites it with a numbered link to a References section at the end of that notebook, so the citation renders without running anything, in JupyterLab, on GitHub, and in the docs:

- In prose, write `[[1]](#ref-1)`, followed by any equation, figure, or appendix number in parentheses outside the link: `... is self-inverse [[1]](#ref-1) (Eq. 41).` Never write a bare "Eq. 21"; always pair it with the reference it belongs to.
- Number the sources `1, 2, ...` in the order each is first cited in that notebook. A source cited again reuses its number.
- End the notebook with a Markdown cell that starts with the heading `## References` and has one numbered entry per source, each starting with its anchor and ending with its DOI link:

  ```markdown
  ## References

  1. <a id="ref-1"></a> N. Baddour, "The Discrete Hankel Transform," in *Fourier Transforms - Century of Digitalization and Increasing Expectations*, IntechOpen, 2019. [doi:10.5772/intechopen.84399](https://doi.org/10.5772/intechopen.84399)
  ```

- Every source a notebook cites must also be listed under `references:` in [`CITATION.cff`](CITATION.cff). When you cite a new paper, add it there too.

`tests/test_notebook_citations.py` checks all of these rules for every notebook under `notebooks/`.

### Notebook style

Tutorial notebooks are read as documentation, top to bottom, so they follow a few more rules:

- Write each Markdown paragraph and each list item on a single source line. Don't wrap lines by hand; the viewer wraps them.
- Render math with `$...$` inline and `$$...$$` for display equations. Backticks are for Python identifiers only, such as `grid.r` or `forward_pft`, never for formulas.
- Give every code cell a short comment per logical step (build the grid, sample, transform, compare, plot), so the cell reads on its own.
- When a cell's point is a shape, a distribution, or an error pattern, plot it; a bare printed number only supplements a figure or serves as a pass/fail check.
- In a notebook that draws figures, start the first code cell with `%matplotlib inline`, after a one-line comment saying it draws each figure below the cell that makes it, so figures appear inline whatever backend the viewer's kernel defaults to.
- Label both axes of every plot that shows a quantity. An image panel whose axes carry no quantity, such as a photograph, may leave them unlabeled, as long as its title says what it shows.
- Describe PyPFT's own conventions only. Don't name the libraries PyPFT uses internally for image resampling, or compare PyPFT's conventions to theirs.

`tests/test_notebook_style.py` checks the first two rules and the `%matplotlib inline` rule, and `tests/test_no_backend_mentions.py` checks the last one, for every notebook under `notebooks/`. `tests/test_no_backend_mentions.py` also checks `README.md`, `CONTRIBUTING.md`, and the pages under `docs/`.

### Spelling

Use American English.

### Code-sectioning comments

Sections are marked with `# ` followed by a sectioning character repeated to the full line width:

- Section: `=`
- Subsection: `-`
- Indented section: `*`
- Indented subsection: `.`

```python
# ========================================================================================
# Section
# ========================================================================================

    # ************************************************************************************
    # Indented section
    # ************************************************************************************

    # ....................................................................................
    # Indented subsection
    # ....................................................................................


# ----------------------------------------------------------------------------------------
# Subsection
# ----------------------------------------------------------------------------------------
```

## Opening a pull request

1. Create a branch from `main` in your fork.
2. Keep the change focused on one topic, with tests for any new behavior, and update the docs and notebooks it affects.
3. Run `./scripts/Invoke-QualityGate.ps1` and make sure it passes.
4. Push your branch and open a pull request against `main`, describing what changed and why. CI runs the same quality gate, plus the tests on every supported Python at both ends of the dependency range.
