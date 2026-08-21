# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying paper.

The package was substantially rewritten from scratch in the `start over` commit (deccca8), which deleted a
full prior layout (`backends/`, `cli/`, `core/`, `dft/`, `dht/`, `fields/`, `grids/`, `idft/`, `io/`, docs,
benchmarks, notebooks, release scripts) and is being rebuilt incrementally, one reviewable phase at a time
per `.local_files/develop_plan.md` (gitignored; not part of the package). So far: `src/pypft/dht/` (the
discrete Hankel transform, four strategy implementations), `src/pypft/utils/validators.py` (shared input
validation), `src/pypft/axes.py` (the axis vocabulary and centered-angular convention),
`src/pypft/geometry.py` (the Cartesian↔polar image bridge), `src/pypft/references.py` (citation machinery),
and a Sphinx docs skeleton (`docs/`) with the first two tutorial notebooks. There is still no angular DFT
subsystem, `PolarGrid`, the actual PFT/IPFT pipeline, domain objects, visualization, or a CLI. Do not assume
any prior architecture, module, or API still exists — check the current file tree before referencing paths
from git history.

## Environment and commands

Dependency management is via **uv** (`uv.lock` is committed); the project targets **Python 3.14
exclusively** (`requires-python = ">=3.14,<3.15"`).

- Install/sync the environment: `uv sync` — run this after every pull/branch switch, since `pyproject.toml`
  changes frequently right now.
- Run the test suite: `uv run pytest` (tests live under `tests/`, mirroring `src/pypft/`'s package layout —
  e.g. `tests/dht/` for `src/pypft/dht/`; top-level modules like `axes.py`/`geometry.py`/`references.py`
  have their tests directly under `tests/`). This also executes and checks every tutorial notebook under
  `notebooks/` via `nbmake`.
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Build the package: `uv build`
- Build the docs: `uv run sphinx-build -W docs docs/_build` (warnings fail the build; `docs/_build/`,
  `docs/_notebooks/`, `docs/jupyter_execute/`, and `docs/.jupyter_cache/` are gitignored, generated
  artifacts — see the docs section below for why `_notebooks/`/`jupyter_execute/` exist at all).
- Add/upgrade a dependency (don't hand-edit version pins): `uv add "<pkg>>=X.Y.Z"` or
  `uv add --group dev "<pkg>>=X.Y.Z"`, then `uv lock` / `uv sync`.
- Format/lint/type-check: `uv run black src tests benchmarks`, `uv run isort src tests benchmarks`,
  `uv run flake8 src` (only `src` is linted — `--extend-select=D1` in `pyproject.toml` enforces
  missing-docstring checks via `flake8-docstrings`), `uv run pyright`, `uv run vulture src`.
- Run the whole quality gate at once (what CI runs): `./scripts/Invoke-QualityGate.ps1` — pytest, black
  `--check`, isort `--check-only`, flake8, pyright, vulture, `sphinx-build -W`, `uv build`, in that order,
  stopping at the first failure. `./scripts/Test-Notebooks.ps1` runs `uv run pytest --nbmake notebooks/`
  separately.
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

## Architecture: the package facade

`src/pypft/__init__.py` re-exports the public surface from each submodule (`Axis`, `DEFAULT_BATCH_AXIS`,
the DHT API, the geometry functions, `Reference`/`cite`/`bibliography`), listed in `__all__` — this is what
keeps `flake8`'s unused-import check (`F401`) satisfied for a pure re-export module.

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
- New validators land here only for shared/third-party types; check the module docstring's "expected
  additions by phase" style inventory in the development plan before adding one, so a validator that
  belongs on a not-yet-written class isn't accidentally duplicated later.

## Architecture: axes and the centered-angular convention (`src/pypft/axes.py`)

`Axis(IntEnum)` names PyPFT's `(radial, angular[, batch])` array axes (`RADIAL = 0`, `ANGULAR = 1`,
`BATCH = 2`). It is an `IntEnum` specifically so `isinstance(Axis.RADIAL, int)` is `True` — every `axis`
parameter typed `Axis | int` is already covered by `IntValidator.type_is_int`, and must **not** be
type-validated with `EnumValidator.type_is_enum` first (that raises on a bare `int`).

Only the batch axis is ever defaulted (`DEFAULT_BATCH_AXIS = -1`), since `-1` and `Axis.BATCH` (`2`) name
the same physical axis on a 3-D `(radial, angular, batch)` array — "default to the last axis" and "default
to the batch axis" coincide exactly where that's unambiguous. Low-level generic transforms (`pypft.dht`)
separately default their own `axis` to `-1` for an unrelated, purely conventional reason; polar-layer
functions never default a *transform* axis.

`_center_angular`/`_uncenter_angular` reorder an angular axis between "natural" order (index `0` holds
angle/harmonic `0`, ascending — what `cv2.warpPolar` and an uncentered DFT both produce) and PyPFT's own
"centered" order (index `i` holds angle/harmonic `i - size // 2`). **`axes.py` is the only module in
`src/` allowed to call `numpy.fft.fftshift`/`ifftshift`** — a lint-as-test in `tests/test_axes.py` asserts
this by scanning every other file under `src/pypft/` for the literal names. Anything that needs to
reorder an angular axis imports these two helpers instead of calling `fftshift`/`ifftshift` directly.

## Architecture: the Cartesian↔polar image bridge (`src/pypft/geometry.py`)

`cartesian_to_polar`/`polar_to_cartesian` wrap `cv2.warpPolar` (forward and its inverse-map mode) to
resample an ordinary image onto, and back off of, a **uniform** polar grid. This is explicitly *not* the
discrete Hankel transform's own sampling grid (which is order-dependent and non-uniform, per Baddour's
`r_nk`) — these two functions exist because `warpPolar` is the natural first illustration of what "polar"
means for an image, not because their output feeds the transform. A real, order-dependent sampler
(`pypft.grid.sample_cartesian`) is future work.

Two deliberate conversions happen at this boundary, each undone by the opposite function:

- **Layout.** `cv2.warpPolar` lays its own output out `(angular, radial[, channel])` — the reference
  implementation's convention. PyPFT's own convention is the opposite, `(radial, angular[, channel])`
  (`pypft.axes.Axis`), so every crossing of this boundary transposes via `numpy.moveaxis`.
- **Angular convention.** `warpPolar`'s angular axis is in natural order; PyPFT's is centered. `Axis` /
  `axes._center_angular`/`_uncenter_angular` handle this, so nothing downstream has to think about it.

One easy-to-get-backwards fact, pinned by `tests/test_geometry.py`'s angular-origin test: `cv2.warpPolar`
measures its angle directly on image coordinates (`atan2(row - center_y, col - center_x)`, no `y`-flip), so
rotating from the positive-`x` axis towards the positive-`y` axis (downward on screen) is its *positive*
direction — counter-clockwise in image coordinates, but clockwise as the image is drawn. `+pi/2` and
`-pi/2` are tested as separate cases because a symmetric test set can't see this sign convention if it's
ever inverted by accident.

## Architecture: citations (`src/pypft/references.py`)

`Reference(Enum)` holds one member per cited scientific source (each an `_Entry` with a `key`, an `inline`
Markdown label, a full reference string, and a DOI); `cite(*refs)` joins inline labels for prose, and
`bibliography(*refs)` renders a sorted Markdown reference list.

This is public API (a scientific package's users legitimately need "how do I cite this?"), but its main
consumer is the tutorial notebooks. Because a notebook's Markdown cell cannot execute code, a citing
sentence writes a member's `inline` label out **literally** (e.g. "...is self-inverse [Baddour2019a, Eq.
41]."); only a notebook's *final* cell — a code cell calling `bibliography(...)` — is actually generated
from this module. `tests/test_notebook_citations.py` machine-checks the two halves of this discipline for
every notebook under `notebooks/`: every bracketed label used in Markdown must resolve to a real `Reference`
member, and any notebook that cites something must end with a `bibliography(...)` call listing exactly what
it cited (no orphans, no omissions).

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
  convention). A third template method, `_apply_along_axis`, generalizes `_apply` to an arbitrary `axis` of
  an N-D array by moving that axis to `-2` (not `0`) before delegating to `_apply` and moving it back
  after — `-2` is load-bearing because `numpy.matmul` (which every `_apply` is built on) treats only the
  *last two* dimensions as the matrix and broadcasts over every leading dimension, so a trailing batch axis
  is only mishandled if the target axis is moved to `0` instead. `forward`/`inverse` take a keyword-only
  `axis: int = -1`.
- **Implementations**, each overriding one hook of the class above it: `_naive.NaiveDHT` (direct
  `scipy.special.jv` calls, `moveaxis`+`matmul`) → `_cached.CachedBesselDHT` (LRU-caches the kernel by
  `(n, size)`, bounded by `KERNEL_CACHE_MAXSIZE`, since it depends only on those two, never on the signal)
  → `_vectorized.VectorizedDHT` (overrides only `_apply`, using `numba`-parallelized loops instead of
  NumPy's BLAS matmul — for `ndim > 2` it flattens the leading dimensions into one batch dimension before
  calling the numba kernel and unflattens after — inheriting `CachedBesselDHT`'s kernel unchanged).
- **`RECURRENCE_BESSEL` was removed (numerically unsound, live-bug fix):** an earlier `_recurrence.py`
  implementation built the kernel's Bessel values via the three-term order recurrence instead of one direct
  `jv()` call per order. That recurrence is exponentially unstable once the order exceeds the argument,
  which the kernel evaluates by construction — `max|Y^{nN} @ Y^{nN} - I|` measured at 2.1e+16 by order 47
  (size 64), against ~7e-6 for the direct/cached kernel at the same order. It was never caught because
  `tests/dht/conftest.py`'s `DHT_ORDERS` used to stop at order 4; `DHT_ORDERS` now includes 16/32/64
  specifically to keep this class of regression visible, and `VectorizedDHT` was reparented from
  `RecurrenceBesselDHT` to `CachedBesselDHT` (it only overrides `_apply`, so reparenting was a pure fix).
- **Order-dependent tolerance:** even the correct kernel's self-inverse residual grows with order (an
  inherent discretization effect, not a bug) — from ~1.9e-9 at order 0 to ~1.1e-5 at order 64 (size 64). A
  flat tolerance is either too loose (hiding a regression) or too tight (rejecting the correct kernel above
  ~order 24), so order-sensitive assertions in `tests/dht/` use `tests/dht/tolerance.py`'s
  `dht_tolerance(order, size)` model instead of the flat `RTOL`/`ATOL` in `tests/dht/conftest.py`.
- **Selection**: `DHTImplementation` (an `Enum`) maps to these three classes via the `_IMPLEMENTATIONS` dict
  in `src/pypft/dht/__init__.py`. `DEFAULT_IMPLEMENTATION` is a hardcoded module-level constant
  (`CACHED_BESSEL`), chosen by running `benchmarks/run_dht_benchmarks.py`: for repeated forward calls at a
  fixed order/size (the realistic usage pattern), `CACHED_BESSEL` is ~2800-3000x faster than `NAIVE`, while
  `VECTORIZED`'s `numba` thread overhead loses to plain BLAS matmul at the benchmarked sizes — re-run that
  script (it exports a timestamped Markdown report to `.local_files/benchmarks/results/`) before changing
  `DEFAULT_IMPLEMENTATION`.
- **Public API**: `hankel_transform`, `inverse_hankel_transform`, and `sample_points` in
  `src/pypft/dht/__init__.py` validate their arguments (via the `*Validator` classes above) and dispatch
  through `_IMPLEMENTATIONS`; internal `BaseDHT` subclass methods assume already-validated input.

## Documentation (`docs/`)

A Sphinx skeleton (`furo` theme, `myst_nb` for notebooks, plain `autodoc` for the API reference — docstrings
are RST, not Google/NumPy style, so no napoleon extension is needed). Built with `sphinx-build -W docs
docs/_build` (warnings fail the build), part of the quality gate from this phase onward.

`notebooks/` is tracked at the repo root, not under `docs/`, so it can be shared as-is with `nbmake` (see
below). Sphinx requires toctree documents to live under its own source directory, so `docs/conf.py`
registers a `config-inited` hook that copies `notebooks/*.ipynb` into `docs/_notebooks/` before Sphinx reads
its sources; `docs/tutorials.rst`'s toctree globs `_notebooks/*`. This avoids a symlink (which needs
elevated privileges on Windows, one of CI's three platforms). `nb_execution_mode = "off"` — notebooks are
executed and checked by `nbmake` in CI, not re-executed by the docs build.

## Notebooks

`notebooks/` is tracked and executed in CI via `nbmake` (see the CI section above), as one incremental
tutorial sequence where each notebook assumes only its predecessors: `00_installation_and_quickstart.ipynb`
and `01_polar_and_cartesian_images.ipynb` exist so far. See "Architecture: citations" above for the
citation discipline notebooks must follow when they state a mathematical result.

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
  double-backtick literals (`` ``Name`` ``) instead, per the existing files. This also matters for
  `sphinx-build -W`: a module docstring written as plain indented prose rather than valid RST (inconsistent
  bullet-list indentation, missing blank lines around nested lists) raises docutils errors that `-W`
  escalates to build failures, even though `flake8-rst-docstrings` may not catch every case in isolation.
- **Line length**: 88 chars everywhere — `pyproject.toml`'s `[tool.black]`/`[tool.flake8]`, `README.md`,
  and `.vscode/settings.json` (editor rulers, `rewrap.wrappingColumn`) all agree.
- **Code-sectioning comments**: `# ` + repeated character to fill the line width — `=` for a top-level
  section, `-` for a subsection, `*`/`.` for the same one indent level in. Snippets: `@ section`,
  `@ subsection`, `@ isection`, `@ isubsection` in `.vscode/helpers.code-snippets`.

## Benchmarking

`benchmarks/` (tracked) holds `bench_dht.py` (pytest-benchmark test functions comparing the DHT
implementations) and `run_dht_benchmarks.py` (runs them and exports a sorted Markdown report); see the DHT
architecture section above for how its results drove `DEFAULT_IMPLEMENTATION`.

## Local, gitignored data

`.local_files/` (gitignored) holds local scratch/reference material for development, not part of the
package or test suite:

- `sources/` — reference papers on the DHT, the polar-coordinate DFT/PFT, and Bessel functions (Baddour
  2019's DHT book chapter and Mathematics Part I paper, Yao & Baddour's PeerJ CS Part II paper and its
  supplementary appendix, and `bessel_properties.md`, a distillation of Bessel-function recurrence/
  derivative relations — retained for reference even though the `_recurrence.py` implementation that used
  it was removed for numerical instability; see the DHT architecture section above). These are the sources
  `src/pypft/references.py`'s `Reference` members cite.
- `benchmarks/results/` — timestamped Markdown reports generated by `benchmarks/run_dht_benchmarks.py`
  (gitignored since they're generated artifacts, not source, even though the scripts that produce them are
  tracked in the top-level `benchmarks/` directory).
- `develop_plan.md` — the phased development plan driving this rebuild (gitignored; not part of the
  package). Each phase is one reviewable PR with its own file list and "do not touch" list.
