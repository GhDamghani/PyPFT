# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying paper.

The package was substantially rewritten from scratch in the `start over` commit (deccca8), which deleted a
full prior layout (`backends/`, `cli/`, `core/`, `dft/`, `dht/`, `fields/`, `grids/`, `idft/`, `io/`, docs,
benchmarks, notebooks, release scripts). It is being rebuilt incrementally. `src/pypft/utils/validators.py`
and `src/pypft/dht/` (the discrete Hankel transform) are the only real code so far; there is still no CLI,
DFT implementation, or documentation beyond `README.md`. Do not assume any prior architecture, module, or
API still exists — check the current file tree before referencing paths from git history.

## Environment and commands

Dependency management is via **uv** (`uv.lock` is committed); the project targets **Python 3.14
exclusively** (`requires-python = ">=3.14,<3.15"`).

- Install/sync the environment: `uv sync` — run this after every pull/branch switch, since `pyproject.toml`
  changes frequently right now.
- Run the test suite: `uv run pytest` (tests live under `tests/`, mirroring `src/pypft/`'s package layout —
  e.g. `tests/dht/` for `src/pypft/dht/`)
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Build the package: `uv build`
- Add/upgrade a dependency (don't hand-edit version pins): `uv add "<pkg>>=X.Y.Z"` or
  `uv add --group dev "<pkg>>=X.Y.Z"`, then `uv lock` / `uv sync`.
- Format/lint/type-check: `uv run black src tests benchmarks`, `uv run isort src tests benchmarks`,
  `uv run flake8 src` (only `src` is linted — `--extend-select=D1` in `pyproject.toml` enforces
  missing-docstring checks via `flake8-docstrings`), `uv run pyright`, `uv run vulture src`.
- Run the whole quality gate at once (what CI runs): `./scripts/Invoke-QualityGate.ps1` — pytest, black
  `--check`, isort `--check-only`, flake8, pyright, vulture, `uv build`, in that order, stopping at the
  first failure. `./scripts/Test-Notebooks.ps1` runs `uv run pytest --nbmake notebooks/` separately and
  treats "no notebooks collected yet" (pytest exit code 5) as a pass, not a failure — real notebook content
  starts arriving later in the project's roadmap.
- Benchmark the DHT implementations: `uv run python benchmarks/run_dht_benchmarks.py` — not part of
  `uv run pytest` (it lives outside `testpaths`), since it's dev tooling, not a correctness check. Writes
  its report to the gitignored `.local_files/benchmarks/results/`, since reports are generated artifacts.
- `pytest.ini_options` sets `filterwarnings = ["error"]` — any `warnings.warn` in `src/` needs a matching
  `pytest.warns` test, or the suite fails.

`pyproject.toml`'s `[project]` table must keep `dependencies = [...]` before `[project.urls]` — TOML
otherwise attaches a bare `dependencies` key to whichever table header precedes it (this previously broke
the build: `numba`/`scipy` silently vanished from the resolved lock and `uv sync` failed with a setuptools
`project.urls.dependencies` validation error).

## CI

`.github/workflows/ci.yml` runs on every pull request and on push to `main`, across a
`windows-latest`/`ubuntu-latest`/`macos-latest` matrix (`shell: pwsh` throughout): `astral-sh/setup-uv`,
`uv sync`, then `scripts/Invoke-QualityGate.ps1` and `scripts/Test-Notebooks.ps1`. No inline shell logic
lives in the YAML — both scripts are meant to be run locally too, so a red CI leg is always reproducible
with one local command. `CHANGELOG.md` follows Keep a Changelog, with changes accumulating under
`## [Unreleased]` until a release.

## Architecture: the validators module

`src/pypft/utils/` is a real (typed) package, not an implicit namespace package — it has an
`__init__.py`. `src/pypft/py.typed` marks the whole distribution as typed (PEP 561); both are declared in
`[tool.setuptools.package-data]` so `uv build` includes `py.typed` in the wheel.

`src/pypft/utils/validators.py` is the shared validation module, and its module docstring is the
authoritative spec for how validation is done project-wide — read it before adding a new validator. Key
points:

- One class per type, named `<Type>Validator` (e.g. `IntValidator`, `PathValidator`), holding
  `@staticmethod`s. Validators for locally-defined (in-package) types live on the class where that type is
  defined, to avoid circular imports — not in this shared module (e.g. `DHTImplementation`'s enum
  membership is validated in `src/pypft/dht/__init__.py`, not here).
- Methods are named `type_is_<typename>` (type-validators, raise `TypeError`) or
  `value_<is|has|should|...>_<condition>` (value-validators, raise `ValueError`, or an `OSError` subclass
  for filesystem-state checks like "path writable").
- Validators only validate — they never mutate/replace their arguments — and assume prior type-validation
  has already run (no redundant `isinstance` re-checks inside a value-validator).
- Classes/methods are ordered to mirror PEP 8 import order: builtins → stdlib → third-party, with
  type-validators before value-validators, simple types before composite types, single-input before
  multi-input validators.
- VS Code snippets in `.vscode/helpers.code-snippets` scaffold new validators: `v-type` for a
  type-validator, `v-value` for a value-validator (the snippets use `@classmethod`; match the file's actual
  `@staticmethod` convention instead).

## Architecture: the discrete Hankel transform (`src/pypft/dht/`)

Implements the DHT as a transform in its own right (not a discretized integral), following N. Baddour,
"The Discrete Hankel Transform" (2019, `.local_files/sources/baddour2019.md`) — read that source before
changing any of the kernel math. Key points:

- **Kernel choice**: the transform uses Baddour's `Y^{nN}` formulation (paper's Eq. 39), not the
  alternative symmetric `T^{nN}` formulation (Eq. 44). Both are self-inverse (`M @ M = I`), but `T^{nN}`
  only preserves Parseval's theorem on Sec. 7's "scaled" vectors, not on raw `f`/`F` values directly —
  `Y^{nN}` was chosen so the public API works in terms of raw signal values. This was found by empirical
  verification against a known continuous Hankel-transform pair (a self-reciprocal Gaussian) after an
  initial `T^{nN}`-based attempt produced wrong (sign-oscillating) results; see `tests/dht/test_gaussian.py`
  for that check and `src/pypft/dht/_base.py`'s module docstring for the full rationale.
- **Strategy pattern**: `src/pypft/dht/_base.py`'s `BaseDHT` defines two overridable hooks —
  `_bessel_kernel(n, size)` (computing the Bessel-valued kernel) and `_apply(kernel, vector)` (applying it
  to a signal) — matching the two independent optimization axes of the transform. `forward`/`inverse` are
  template methods built on top of those hooks and never need overriding. All methods are
  classmethods/staticmethods; a DHT implementation is never instantiated (mirrors the validators' stateless
  convention).
- **Implementations**, each overriding one hook of the class above it: `_naive.NaiveDHT` (direct
  `scipy.special.jv` calls, plain matmul) → `_cached.CachedBesselDHT` (LRU-caches the kernel by `(n, size)`,
  since it depends only on those two, never on the signal) → `_recurrence.RecurrenceBesselDHT` (builds the
  kernel's Bessel values via the three-term recurrence from `.local_files/sources/bessel_properties.md`
  instead of one direct `jv()` call per order, still cached) → `_vectorized.VectorizedDHT` (overrides only
  `_apply`, using `numba`-parallelized loops instead of NumPy's BLAS matmul).
- **Selection**: `DHTImplementation` (an `Enum`) maps to these four classes via the `_IMPLEMENTATIONS` dict
  in `src/pypft/dht/__init__.py`. `DEFAULT_IMPLEMENTATION` is a hardcoded module-level constant, chosen by
  running `benchmarks/run_dht_benchmarks.py`: for repeated forward calls at a fixed order/size
  (the realistic usage pattern), `RECURRENCE_BESSEL` and `CACHED_BESSEL` are ~3000x faster than `NAIVE` and
  statistically tied with each other, while `VECTORIZED`'s `numba` thread overhead actually loses to plain
  BLAS matmul at the benchmarked sizes — re-run that script (it exports a timestamped Markdown report to
  `.local_files/benchmarks/results/`) before changing `DEFAULT_IMPLEMENTATION`.
- **Public API**: `hankel_transform`, `inverse_hankel_transform`, and `sample_points` in
  `src/pypft/dht/__init__.py` validate their arguments (via the `*Validator` classes above) and dispatch
  through `_IMPLEMENTATIONS`; internal `BaseDHT` subclass methods assume already-validated input.

## Conventions (from `README.md`)

- **Constant extraction**: extract constants as much as possible — module-level constants go at the top of
  the module; anything shared more broadly goes at whatever level actually shares it.
- **Errors/warnings**: raise exceptions for invalid state — prefer purpose-built subclasses of builtin
  exceptions (e.g. `Val1ValueError(ValueError)`) over generic ones when callers need to distinguish cases;
  use the `warnings` module (not silent fallback) when degrading gracefully instead of raising.
- **Type annotations**: required on all function signatures, with input validation via the
  `*Validator` classes above.
- **Docstrings**: reStructuredText (PEP 287 / Sphinx `:param:`/`:type:`/`:raises:` style), not
  Google/NumPy style — see any method in `validators.py` for the pattern. Avoid Sphinx cross-reference
  roles (`:class:`/`:func:`/`:meth:`/`:data:`) — flake8's RST checker doesn't recognize them; use plain
  double-backtick literals (`` ``Name`` ``) instead, per the existing files.
- **Line length**: 88 chars everywhere — `pyproject.toml`'s `[tool.black]`/`[tool.flake8]`, `README.md`,
  and `.vscode/settings.json` (editor rulers, `rewrap.wrappingColumn`) all agree.
- **Code-sectioning comments**: `# ` + repeated character to fill the line width — `=` for a top-level
  section, `-` for a subsection, `*`/`.` for the same one indent level in. Snippets: `@ section`,
  `@ subsection`, `@ isection`, `@ isubsection` in `.vscode/helpers.code-snippets`.

## Benchmarking

`benchmarks/` (tracked) holds `bench_dht.py` (pytest-benchmark test functions comparing the DHT
implementations) and `run_dht_benchmarks.py` (runs them and exports a sorted Markdown report); see the DHT
architecture section above for how its results drove `DEFAULT_IMPLEMENTATION`.

## Notebooks

`notebooks/` is tracked and executed in CI via `nbmake` (see the CI section above), as one incremental
tutorial sequence where each notebook assumes only its predecessors. It is currently empty — tutorial
content lands alongside the features it demonstrates.

## Local, gitignored data

`.local_files/` (gitignored) holds local scratch/reference material for development, not part of the
package or test suite:

- `sources/` — reference papers on the DHT and Bessel functions (Baddour 2019, an MDPI "mathematics" paper,
  a PeerJ CS paper, and `bessel_properties.md`, a distillation of Bessel-function recurrence/derivative
  relations used by `src/pypft/dht/_recurrence.py`).
- `benchmarks/results/` — timestamped Markdown reports generated by `benchmarks/run_dht_benchmarks.py`
  (gitignored since they're generated artifacts, not source, even though the scripts that produce them are
  tracked in the top-level `benchmarks/` directory).
