# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying papers. `README.md` is a short landing page (description, pipeline equation, install, quickstart, citing, license); the per-layer API tour lives in `docs/user_guide.rst`, and contributor material (dev setup, quality gate, conventions, PR process) lives in `CONTRIBUTING.md` plus the `docs/contributing.rst` setup tutorial. `CLAUDE.md` itself is tracked; `.vscode/` is gitignored and untracked -- no tracked file may depend on it (the snippets this file names are a convenience, never required).

PyPFT is built incrementally, one reviewable unit at a time. So far: `src/pypft/dht/` (the
discrete Hankel transform, three strategy implementations), `src/pypft/dft/` (the angular discrete Fourier
transform, two strategy implementations), `src/pypft/utils/validators.py` (shared input validation),
`src/pypft/axes.py` (the axis vocabulary and centered-angular convention), `src/pypft/geometry.py` (the
Cartesian↔polar image bridge), `src/pypft/grid.py` (`PolarGrid`, the transform's own order-dependent
sampling grid, plus the production `sample_cartesian` sampler, the `resample_uniform_polar`
uniform-polar-to-grid bridge, and the `check_adequacy`/`check_nyquist_adequacy` warnings), `src/pypft/transform.py`
(`forward_pft`/`inverse_pft`, the full PFT/IPFT pipeline, the `scaled_hankel` step underlying both -- two
strategy implementations, accepting a single `(radial, angular)` sample or a `(radial, angular, batch)`
batch of them),
`src/pypft/domains.py` (`PolarDomain`/`BaseSignal`, the typed legal-move shell over `transform.py`'s numerics),
`src/pypft/ring.py` (the ring-consistent route, the accurate but only approximately invertible alternative
to the exact path), `src/pypft/_kernel.py` (a private, from-scratch `O(N**4)` oracle reproducing `forward_pft`/`inverse_pft`,
used only by `tests/test_kernel.py`), `src/pypft/acquisition/` (experimental radial k-space
converted into PyPFT's uniform polar layout: per-array-layout readers, a selection of a single sample or a
batch, and a minimal Textual terminal interface), `src/pypft/cli/` (PyPFT's own Textual terminal
interface: file -> grid -> PFT/IPFT -> view/save figures), `src/pypft/samples/` (downloads test samples
too large to commit from their public sources), `src/pypft/viz.py` (`plot_signal`/`BaseSignal.plot`,
`render_cartesian`, and `forward_pft_traced`/`inverse_pft_traced` -- `Axes`/`Figure`-based visualization,
never writing to disk), analytical-property test suites for both transforms
(`tests/dht/test_kernel_properties.py`, `tests/test_transform_properties.py`), and a Sphinx docs skeleton
(`docs/`) with eleven tutorial notebooks. Command-line entry points: `python -m pypft` / the `pypft`
console script (`[project.scripts] pypft = "pypft.cli:main"`), `python -m pypft.acquisition` and
`python -m pypft.samples` (no console scripts). Do not
assume any prior architecture, module, or API still exists — check the current file tree before referencing
paths from git history; module names like `backends/`, `cli/`, `core/`, `fields/`, `grids/`, `idft/`, `io/`
may still appear in older commits but do not correspond to anything in the current tree. `CHANGELOG.md`
does not exist and must not be recreated without checking with the developer first, even though some commit
messages still follow the Keep-a-Changelog convention. `DESIGN_NOTES.md` (repo root) holds general
technical rationale and pitfall warnings referenced by file/section from source and from this file's own
Architecture sections below — see the Conventions section for what belongs there.

## Environment and commands

Dependency management is via **uv** (`uv.lock` is committed). PyPFT supports every CPython in
`supported_python_versions.txt` (3.12, 3.13, 3.14; `requires-python = ">=3.12"`, no upper cap) and is
**developed on the oldest**: `.python-version` pins `3.12`. `[project] dependencies` publish
per-interpreter wheel floors via `python_version` markers (oldest release with a cp3XX wheel on all three
OSes; the table lives in `pyproject.toml`'s comments), no upper bounds; `[tool.uv]
constraint-dependencies` pins the dev environment to exactly those floors (uv-only, never in wheel
metadata). matplotlib's 3.12 floor is `3.8.1`, not the first-cp312-wheel `3.7.3`, because pyright needs its
inline type hints. `pytest`'s `filterwarnings` also ignores pyparsing-3.3's camelCase deprecation warning,
which every matplotlib before 3.11 triggers internally. Code must stay 3.12-compatible (no PEP 649
unquoted forward references, no 3.13+ stdlib APIs). To test the newest releases locally:
`uv pip install --no-config --upgrade matplotlib numba llvmlite numpy opencv-python-headless rich scipy textual textual-fspicker`, then
`./scripts/Invoke-QualityGate.ps1 -NoSync`.

- Install/sync the environment: `uv sync` — run this after every pull/branch switch, since `pyproject.toml`
  changes frequently right now.
- Run the test suite: `uv run pytest` (tests live under `tests/`, mirroring `src/pypft/`'s package layout —
  e.g. `tests/dht/` for `src/pypft/dht/`; top-level modules like `axes.py`/`geometry.py`/`grid.py`
  have their tests directly under `tests/`). This also executes and checks every tutorial notebook under
  `notebooks/` via `nbmake`.
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Build the package: `uv build`
- Build the docs: `uv run sphinx-build -W docs docs/_build` (warnings fail the build; `docs/_build/`,
  `docs/_notebooks/`, `docs/jupyter_execute/`, and `docs/.jupyter_cache/` are gitignored, generated
  artifacts — see the docs section below for why `_notebooks/`/`jupyter_execute/` exist at all).
- Add/upgrade a dependency (don't hand-edit version pins): `uv add "<pkg>>=X.Y.Z"` or
  `uv add --group dev "<pkg>>=X.Y.Z"`, then `uv lock` / `uv sync`.
- Format/lint/type-check: `uv run black src tests benchmarks scripts`,
  `uv run isort src tests benchmarks scripts`, `uv run flake8 src scripts` (only `src` and `scripts` are
  linted — `--extend-select=D1` in `pyproject.toml` enforces missing-docstring checks via
  `flake8-docstrings`), `uv run pyright`, `uv run vulture src scripts`.
- Run the whole quality gate at once (what CI runs): `./scripts/Invoke-QualityGate.ps1` — fetching
  missing test samples (`python -m pypft.samples --directory tests/samples --cache-directory
  .local_files/samples`), pytest, the
  notebook suite (`./scripts/Test-Notebooks.ps1`, `uv run pytest --nbmake notebooks/`), black `--check`,
  isort `--check-only`, flake8, pyright, vulture, `sphinx-build -W`, `uv build`, in that order, stopping at
  the first failure. `Test-Notebooks.ps1` is still a separate, standalone script (useful on its own when
  only notebooks changed), just also invoked as one step of the main gate rather than only by CI calling it
  a second time.
- Benchmark the DHT/DFT/PFT-batching implementations: `uv run python benchmarks/run_dht_benchmarks.py` /
  `uv run python benchmarks/run_dft_benchmarks.py` / `uv run python benchmarks/run_pft_benchmarks.py` —
  none are part of `uv run pytest` (they live outside `testpaths`), since they're dev tooling, not a
  correctness check. All three write timestamped Markdown reports to the gitignored
  `.local_files/benchmarks/results/`.
- Regenerate the package's Mermaid class diagram: `uv run python scripts/make_class_diagram.py
  [--output <path>]` — parses `src/pypft` with `ast` (nothing is imported, stdlib only), printing to
  stdout by default. `--no-private` gives the public-API view, `--no-module-functions` the pure-class
  view. Dev tooling like the benchmarks: not part of `uv run pytest` or the quality gate, and the
  diagram is not committed anywhere, so nothing goes stale when `src/` changes.
- Regenerate the maze test fixture: `uv run python scripts/make_maze.py` (defaults `--rings 8`,
  `--inner-sectors 8`, `--seed 0`, `--wall-thickness 0.25`, `--n-radial 576`, `--n-angular 48`, `--radius
  1.0`, `--maze-fraction 0.95`, `--image-size 512`, `--output-dir tests/samples`) — a **circular** maze
  built in polar coordinates: a courtyard inside concentric rings of cells, sector counts doubling outward
  but capped at `n_angular // SAMPLES_PER_SECTOR` (3) and always dividing `n_angular`, so every radial wall
  sits exactly on a sample angle (default layout `(8,) + (16,) * 7`). Carved by iterative randomized DFS
  from the outer entrance, driven only by `numpy.random.default_rng(seed)`, plus one courtyard doorway.
  Walls (value `0`, uniform physical thickness) are evaluated analytically by `draw_maze` —
  **no Cartesian raster and no `sample_cartesian`**, which is the point: a polar-native image — through
  two samplers: `sample_maze_on_grid` (at `grid.r`/`grid.theta`, the exact path) and
  `sample_maze_uniform_polar` (uniform radii `k * R / n_radial`, `cartesian_to_polar`'s convention, same
  spokes and counts: "what your data is on"). Writes `maze_polar.tif` (the fixture: uncompressed 8-bit
  `(radial, angular)` TIFF on `PolarGrid`, the `PolarDomain.SPATIAL_ANGULAR` sample signal in
  `notebooks/08_visualization.ipynb`, loaded with `plt.imread`, feeds `forward_pft` directly),
  `maze_uniform_polar.tif` (same format, goes through `resample_uniform_polar` first; its `forward_pft`
  is within 7.8e-3 relative L2 of the on-grid fixture's), `maze_cartesian.png` (`render_cartesian`'s
  display) and `maze_source.png` (the same maze evaluated on a pixel grid). A third sampler,
  `sample_maze_rings(maze, *, grid, ..., n_quadrature=DEFAULT_RING_QUADRATURE)` (1024 angles), returns the
  maze's complex spatial harmonics on each harmonic's own true rings (the ring route's input, `[0, 255]`
  scale) and writes **no** file: `maze_polar.tif` stays the only maze fixture. At the default 576x48 grid
  it takes ~18s. The default grid triggers
  `check_adequacy`'s `AdequacyWarning` (predicted 0.43 > 0.25): expected, since the grid is fixed by the
  committed fixture, and `tests/test_make_maze.py` wraps `main()` in `pytest.warns`. The grid
  size is bounded by `jn_zeros` cost: 576x48 takes ~8s, 784x96 ~70s. The defaults reproduce the
  committed files; `tests/test_make_maze.py` (which imports the script via `importlib`, since `scripts/` is
  not a package) checks that array-for-array, plus seed determinism and maze perfection. Neither
  `THIRD-PARTY-NOTICES.md` nor `scripts/make_test_image.py` exists in the tree.
- Download the test samples that are not in the repository: `uv run python -m pypft.samples --directory
  tests/samples --cache-directory .local_files/samples` (samples already present are skipped; `--overwrite`
  rewrites them). The only one so far is `tests/samples/P01_slc1_phs1_lins196_cols416_cha4.mat`, the MD-CNN
  excerpt that `tests/acquisition`, `tests/cli` and `notebooks/09_experimental_data.ipynb` read; it is
  gitignored and **must never be committed** (it was purged from history). Without it those tests fail, so
  run this once per clone (the gate does it for you).
- Run the acquisition converter's terminal interface: `uv run python -m pypft.acquisition [path]`.
- Run PyPFT's terminal interface: `uv run pypft` (or `uv run python -m pypft`); the maze fixture
  `tests/samples/maze_polar.tif` is on its own grid with `R = 1`.
- Strip notebooks to their committed form: `python scripts/strip_notebook.py [--check] [paths]` (defaults
  to `notebooks/`). Standard-library only, so it runs without the project environment; it is what the
  `strip-notebook` pre-commit hook (`.pre-commit-config.yaml`, installed per clone with
  `uv run pre-commit install`) calls, and `tests/test_notebook_clean.py` checks the same form. It drops
  notebook/cell metadata, outputs and execution counts but **renumbers cell ids to `"0"`, `"1"`, ...
  rather than dropping them**: nbformat 4.5 requires ids, and a notebook without them raises
  `MissingIDFieldWarning`, which fails `nbmake` under `filterwarnings = ["error"]`. Opening a notebook in
  VS Code re-adds a `kernelspec`/`language_info` block, which fails CI if committed unstripped.
- `pytest.ini_options` sets `filterwarnings = ["error"]` — any `warnings.warn` in `src/` needs a matching
  `pytest.warns` test, or the suite fails.

`pyproject.toml`'s `[project]` table must keep `dependencies = [...]` before `[project.urls]` — see
`DESIGN_NOTES.md`, "Packaging: `dependencies` must precede `[project.urls]`" for why.

## CI

Two workflows, both `ubuntu-latest` only, `shell: pwsh`, on every pull request and push to `main`:

- `.github/workflows/ci.yml` (**quality gate**): one job on the development setup -- `setup-uv` with no
  `python-version`, so uv reads `.python-version` (3.12), `uv sync` to the dependency floors, then
  `scripts/Invoke-QualityGate.ps1` (which runs `scripts/Test-Notebooks.ps1` as one of its own steps).
  Both workflows restore the downloaded test samples with `actions/cache` (key: a hash of
  `src/pypft/samples/**`), so the ~230 MB MD-CNN source is downloaded again only when that module
  changes.
- `.github/workflows/tests.yml` (**tests**): a `python-versions` job reads `supported_python_versions.txt`
  via `scripts/Get-PythonVersions.ps1`, then a Python × `{minimum, latest}` matrix (`UV_PYTHON` per leg):
  `uv sync`, `python -m pypft.samples --directory tests/samples`, on `latest` `uv pip install --no-config --upgrade <runtime deps>`, then
  `uv run --no-sync pytest` and `scripts/Test-Notebooks.ps1 -NoSync` (`--no-sync` keeps the upgrade).
  `Invoke-QualityGate.ps1 -NoSync` still exists for running the whole gate on latest locally.

No inline shell logic lives in the YAML, so a red CI leg is always reproducible locally.

## Architecture: the package facade

`src/pypft/__init__.py` re-exports the public surface from each submodule (`PolarAxis`, `POLAR_SAMPLE_NDIM`, `DEFAULT_BATCH_AXIS`,
the DHT API, the domains API — `PolarDomain`, `BaseSignal`, and its four subclasses — the geometry functions,
the grid API — `PolarGrid`, `LimitKind`, `sample_cartesian`, `resample_uniform_polar`, `check_adequacy`,
`check_nyquist_adequacy` —
`forward_pft`/`inverse_pft`, `resample_uniform_polar_frequency`, the ring-route API — `sample_harmonics_cartesian`,
`sample_harmonics_uniform_polar`, `evaluate_frequency`, `evaluate_space`, `forward_pft_ring`,
`inverse_pft_ring` — and the visualization API —
`PFTTrace`, `forward_pft_traced`, `inverse_pft_traced`, `plot_signal`, `render_cartesian` — listed
in `__all__` — this is what
keeps `flake8`'s unused-import check (`F401`) satisfied for a pure re-export module. `pypft.dft` is deliberately **not** re-exported here: it is internal plumbing
between the geometry/axes layer and the DHT (no notebook of its own — see the Notebooks section), reachable
as `pypft.dft.angular_dft` and documented via `docs/api.rst`, the same way `pypft.utils.validators` is
public but un-re-exported. `pypft.grid`'s two warning classes (`AdequacyWarning`, `NyquistWarning`) and
`pypft.transform`'s `Direction`/`scaled_hankel` are likewise reachable via `pypft.grid.*`/`pypft.transform.*`
but not re-exported at the top level; `pypft.acquisition` is likewise a subpackage reached as
`pypft.acquisition.*` (and its `app` module imports `textual`, so nothing imports it implicitly),
`pypft.samples` (network-touching test tooling) likewise, and
`pypft.cli` (an entry point, also importing `textual`) is never re-exported either: the
warnings because filtering on them is an opt-in developer action
rather than everyday API surface, and `scaled_hankel`/`Direction` because they are `forward_pft`/
`inverse_pft`'s own internal plumbing, the same way `pypft.dft` is internal to the polar layers above it.

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
  membership is validated in `src/pypft/dht/__init__.py`, not here; `PolarGrid`'s own type-validator,
  `_type_is_polar_grid`, likewise lives in `src/pypft/grid.py`; `BaseSignal`'s own type-validator,
  `_type_is_base_signal`, lives in `src/pypft/domains.py` and is reused by `pypft.viz`).
  `NumpyValidator` has `value_is_1d`/`value_is_at_least_1d`, `value_is_2d`/`value_is_finite` (for
  `pypft.grid.sample_cartesian`'s image argument), `value1_axis_length_matches_value2` (single-axis) and
  `value1_shape_matches_value2` (for `pypft.transform.forward_pft`/`inverse_pft`'s whole-array shape check
  against a `PolarGrid`), and `value_has_ndim_in(value, ndims)` (general-purpose rank check). The polar
  layer's "a single sample or a batch" rank check is `pypft.axes._value_is_polar_sample_or_batch` (built
  on `value_has_ndim_in` with `POLAR_SAMPLE_NDIM`), shared by `forward_pft`/`inverse_pft`,
  `pypft.domains.BaseSignal`, `resample_uniform_polar` and `pypft.ring`; `scaled_hankel` accepts any rank
  (`value_is_at_least_1d` plus its two distinct named axes). `MatplotlibValidator` (`type_is_axes`, `type_is_figure`) covers
  `pypft.viz`'s `Axes`/`Figure` arguments and its `PFTTrace.figures` field, ordered before `NumpyValidator`
  (third-party classes are alphabetical: `matplotlib` before `numpy`). `IntValidator.value1_is_less_than_value2`,
  `StrValidator.type_is_str` and `MappingValidator.type_is_mapping` (`collections.abc.Mapping`) validate
  `pypft.acquisition`'s selections; `PathValidator.value_has_suffix_in` (case-insensitive suffix set)
  validates `pypft.cli`'s input files. `tests/utils/test_validators.py` covers those four only.
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

`PolarAxis(IntEnum)` names the two sample axes of a polar array (`RADIAL = 0`, `ANGULAR = 1`) and has
**no batch member**: the batch axis is not a coordinate of the sample but "the axis after the sample axes",
index `POLAR_SAMPLE_NDIM` (`= len(PolarAxis) = 2`), named only by `DEFAULT_BATCH_AXIS`. "3-D" never means
"batched" anywhere in the repo (future spherical/cylindrical samples are genuinely 3-D); prose says "a
single sample" / "a batch of samples". `_value_is_polar_sample_or_batch(value)` is the shared rank
validator, raising "value must be a single sample (radial, angular) or a batch (radial, angular,
batch)". It is an `IntEnum` specifically so `isinstance(PolarAxis.RADIAL, int)` is `True` — every `axis`
parameter typed `PolarAxis | int` is already covered by `IntValidator.type_is_int`, and must **not** be
type-validated with `EnumValidator.type_is_enum` first (that raises on a bare `int`).

Only the batch axis is ever defaulted (`DEFAULT_BATCH_AXIS = -1`), since on a batch the axis after the
sample axes is also the last one — "default to the last axis" and "default to the batch axis" coincide. Low-level generic transforms (`pypft.dht`,
`pypft.dft`) separately default their own `axis` to `-1` for an unrelated, purely conventional reason;
polar-layer functions never default a *transform* axis.

`_center_angular`/`_uncenter_angular` reorder an angular axis between "natural" order (index `0` holds
angle/harmonic `0`, ascending — what `cv2.warpPolar` and an uncentered DFT both produce) and PyPFT's own
"centered" order (index `i` holds angle/harmonic `i - size // 2`). **`axes.py` is the only module in
`src/` allowed to call `numpy.fft.fftshift`/`ifftshift`** — a lint-as-test in `tests/test_axes.py` asserts
this by scanning every other file under `src/pypft/` for the literal names (including inside docstrings —
`src/pypft/dft/_base.py` describes its own centering in prose rather than naming the functions, to stay
clean of the regex). Anything that needs to reorder an angular axis imports these two helpers instead of
calling `fftshift`/`ifftshift` directly — `src/pypft/geometry.py` and `src/pypft/dft/_base.py` are the two
current consumers.

## Architecture: the Cartesian↔polar image bridge (`src/pypft/geometry.py`)

`cartesian_to_polar`/`polar_to_cartesian` wrap `cv2.warpPolar` (forward and its inverse-map mode; no interpolation flag is passed, so `warpPolar` samples the nearest pixel, up to half a pixel off) to
resample an ordinary image onto, and back off of, a **uniform** polar grid. This is explicitly *not* the
discrete Hankel transform's own sampling grid (which is order-dependent and non-uniform, per Baddour's
`r_nk`) — these two functions exist because `warpPolar` is the natural first illustration of what "polar"
means for an image, not because their output feeds the transform. The real, order-dependent sampler is
`pypft.grid.sample_cartesian` (see the sampling-grid section below).

Two deliberate conversions happen at this boundary, each undone by the opposite function:

- **Layout.** `cv2.warpPolar` lays its own output out `(angular, radial[, channel])` — the reference
  implementation's convention. PyPFT's own convention is the opposite, `(radial, angular[, channel])`
  (`pypft.axes.PolarAxis`), so every crossing of this boundary transposes via `numpy.moveaxis`.
- **Angular convention.** `warpPolar`'s angular axis is in natural order; PyPFT's is centered. `PolarAxis` /
  `axes._center_angular`/`_uncenter_angular` handle this, so nothing downstream has to think about it.

One easy-to-get-backwards fact, pinned by `tests/test_geometry.py`'s angular-origin test: `cv2.warpPolar`
measures its angle directly on image coordinates (`atan2(row - center_y, col - center_x)`, no `y`-flip), so
rotating from the positive-`x` axis towards the positive-`y` axis (downward on screen) is its *positive*
direction — counter-clockwise in image coordinates, but clockwise as the image is drawn. `+pi/2` and
`-pi/2` are tested as separate cases because a symmetric test set can't see this sign convention if it's
ever inverted by accident.

## Architecture: the sampling grid (`src/pypft/grid.py`)

`PolarGrid` is the discrete Hankel transform's *actual* sampling grid — order-dependent and
non-uniform, unlike `pypft.geometry`'s uniform illustration grid. It is a frozen, hashable dataclass
(`n_radial`, `n_angular`, `R`, `limit_kind`); every array-valued attribute (`r`, `rho`, `theta`, `psi`,
`harmonics`, `parity`) is a `@property` recomputed from those four fields on access rather than cached on
the instance, since the underlying `pypft.dht.sample_points` call is already memoized by its own
`(order, size)` kernel cache. Key points:

- **Row `i`'s Bessel order is `abs(harmonics(n_angular)[i])`.** `PolarGrid.r`/`.rho` loop over
  `pypft.dft.harmonics(n_angular)` and call `pypft.dht.sample_points` once per row — the "key cross-check"
  this reuses is that `grid.r` at the harmonic-0 row is bit-identical to `sample_points(0, n_radial, R)[0]`,
  since it is the exact same call. `theta`/`psi` are the single 1-D array
  `harmonics(n_angular) * (2 * pi / n_angular)`, shared by both domains.
- **In the space domain, row `i` is a spoke, not a harmonic.** Its radii use the Bessel order of the
  spoke's own angular sample index `p`; the transform *identifies* `p` with the harmonic order (Part I,
  Eq. 18; Appendix A.6), which is what makes it square and invertible. Consequences, all measured in
  `DESIGN_NOTES.md`, "Grid: the spatial row index is a spoke, and the transform identifies it with a
  harmonic," and pinned by `tests/test_grid_relationship.py`: a column of constant radial index is not a
  ring; a uniform polar array (`cartesian_to_polar`) has the same spokes but must be interpolated along
  each spoke onto `grid.r` before transforming (`resample_uniform_polar`, below, reproduces on-grid results); even
  on-grid, the identification leaves a relative L2 error of ~0.14-2.2 that `E_avg` hides (e.g. −98 dB vs.
  L2 2.16). **Every accuracy report gives the relative L2 error next to `E_avg`**, and `check_adequacy`
  is necessary, not sufficient.
- **`LimitKind.SPACE_LIMITED` vs. `BAND_LIMITED`** (PeerJ CS Part II, Eqs. 14-17) are the same
  Bessel-zero ratios with `r`/`rho` swapped and `R` reinterpreted as the band limit `Wr` — implemented as
  exactly that: `PolarGrid.r`/`.rho` pick one of `sample_points`'s two outputs for `SPACE_LIMITED` and the
  other for `BAND_LIMITED`, rather than a second formula.
- **`check_adequacy(grid)`** warns (`AdequacyWarning`, never raises) when `n_radial` is too small for
  `n_angular`, using a log-log least-squares fit of this package's own measured **relative L2 error** of
  the centered forward Gaussian to `(n_angular, n_radial)` at `R=40` (nine points, `n_angular in (15, 32,
  64)` × `n_radial in (383, 767, 1535)`, largest residual a factor of 1.11) — *not* a formula from either
  paper, and blind to `R` and to off-center content. Threshold `0.25`: the worked example `(382, 15, R=40)`
  measures 0.243 (predicted 0.219), so a grid warns when predicted worse than that reference. Fit points,
  coefficients and rationale: `DESIGN_NOTES.md`, "Grid: `check_adequacy` is fitted on the relative L2
  error"; `tests/test_grid.py` re-measures three points against the fit.
- **`resample_uniform_polar(values, grid, *, radius)`** is the bridge from uniform polar data onto the grid:
  `values` is `(radial, angular[, batch])`, real or complex, centered angular axis, radii exactly
  `cartesian_to_polar`'s (`k * radius / n`, `k = 0 .. n-1`; `radius = min(height, width) / 2` for a
  `cartesian_to_polar` result). Same spoke count as the grid → one `scipy` `CubicSpline` per spoke along the
  radius (the outermost piece extrapolates the < 1-step gap past the last sample); different count → a
  radial spline plus a periodic angular spline. Raises `RadiusCoverageError(ValueError)` when
  `grid.R > radius`, `NotImplementedError` for band-limited grids, and warns
  `AngularUpsamplingWarning(UserWarning)` when the data has fewer spokes than the grid. Both classes are
  reachable via `pypft.grid.*` only, like `AdequacyWarning`. Tests: `tests/test_resample.py`. Its input
  checks live in the private `_validate_uniform_polar(values, grid, radius, *, caller)`, shared with
  `pypft.ring`'s uniform samplers so all accept exactly the same input; its `extent`/`extent_name`
  keywords name what must fit inside `radius` (`grid.R` on the space side, `max(grid.rho)` on the
  frequency side). Its required keyword `stacklevel` is 4 everywhere: both public resamplers go through
  `_resample_uniform_polar_onto`, and every ring entry point through `_sample_harmonics_uniform_polar`, so
  the `AngularUpsamplingWarning` points at the user's line on every path — pinned by
  `tests/test_ring.py::test_angular_upsampling_warning_points_at_the_callers_line`.
- **`resample_uniform_polar_frequency(values, grid, *, radius)`** is the frequency-domain counterpart (uniform
  polar k-space, `radius` in radians per unit of `grid.R`) onto `grid.rho`/`grid.psi`: `inverse_pft`'s input.
  The private helpers `_resample_matching_spokes`/`_resample_other_spokes` take the target radii
  (`grid.r` or `grid.rho`) rather than the grid's own space radii. High-order spokes reach farthest:
  `max(grid.rho) ~= (n_radial + n_angular / 4) * pi / R` must not exceed `radius`.
- **`check_nyquist_adequacy(grid, band_limit)`** warns (`NyquistWarning`) using PeerJ CS Part II's Eq. 21
  directly (`j_(0, N1) >= band_limit * R`, checked via `scipy.special.jn_zeros`) — the order-0 zero is the
  binding constraint since it is the smallest across every harmonic order the transform uses. Only
  `LimitKind.SPACE_LIMITED` is supported; band-limited grids raise `NotImplementedError` rather than
  silently checking the wrong condition.
- **`sample_cartesian(image, grid)`** is the production sampler: one `cv2.remap` call at `grid.r`/
  `grid.theta`, using the same angle convention as `pypft.geometry` (measured directly on image
  coordinates, no `y`-flip). `cv2.remap`'s `INTER_LINEAR` interpolates on a fixed-point, 1/32-pixel
  sub-pixel grid rather than at full `float64` precision — `tests/test_grid.py`'s tolerance for this
  function matches `tests/test_geometry.py`'s own round-trip tolerance for the same reason.
- Grid construction is deliberately a **single implementation**, no `DHTImplementation`-style strategy
  pattern or benchmark — there is nothing here to pick a fastest strategy between.

## Architecture: the PFT/IPFT pipeline (`src/pypft/transform.py`)

`forward_pft(f, grid)`/`inverse_pft(F, grid)` are the full chain from `README.md`'s diagram: an angular
DFT/IDFT (`pypft.dft`) around a per-harmonic, `R`-scaled discrete Hankel transform (`pypft.dht`). This is a
direct, verified port of Yao & Baddour's own MATLAB appendix (PeerJ CS Part II, Appendix A-5/A-6) onto
`pypft.dft`'s and `pypft.dht`'s existing, separately-verified public entry points — no new numerical kernel
is introduced here, only composition. Key points:

- **`forward_pft`/`inverse_pft` follow PyPFT's own `(radial, angular[, batch])` axis layout**
  (`pypft.axes.PolarAxis`), matching `pypft.geometry.cartesian_to_polar`'s convention — **not**
  `PolarGrid.r`'s/`sample_cartesian`'s own `(angular, radial)` layout (each row of `PolarGrid.r` is one
  harmonic's own radial samples, the natural shape for *building* the grid, not for storing a transformed
  image). A `sample_cartesian` result must be transposed before it is passed to `forward_pft`. Both accept
  a single sample or a batch of samples — costs no new numerical kernel, since the angular DFT/IDFT and the
  DHT already operate along one named axis of an otherwise arbitrary-rank array. `batch_axis: int =
  DEFAULT_BATCH_AXIS` is keyword-only on both (`pypft.axes`'s axis-default tiering: only the batch axis is
  ever defaulted); passing it on a single sample, or naming anything other than index `POLAR_SAMPLE_NDIM`
  of a batch, raises `ValueError`. The fixed layout is what makes `forward_pft`/`inverse_pft` polar.
- **The steps under them are axis-flexible**, so a future cylindrical/spherical layer can reuse them:
  `scaled_hankel`, `angular_dft`/`inverse_angular_dft` and `hankel_transform` take explicit axes and any
  rank. `_scaled_hankel_stacked_kernel` moves the harmonic/radial axes to `_STACK_HARMONIC_AXIS`/
  `_STACK_RADIAL_AXIS` (0/1) and flattens every other axis into one matmul column axis.
  `tests/test_transform.py`'s "Axis flexibility" section pins this on a 4-D `(3, radial, 2, angular)`
  layout for both implementations and directions.
- **`scaled_hankel(values, grid, *, direction, axis, angular_axis, implementation)`** is the per-harmonic
  step underlying both directions: it loops over `grid.harmonics` (one Hankel transform per harmonic, since
  each order needs its own kernel), and is the single place in `src/` that names the radial axis —
  `forward_pft`/`inverse_pft` always pass `PolarAxis.RADIAL`/`PolarAxis.ANGULAR` explicitly rather than relying on
  `hankel_transform`'s own unrelated `axis=-1` default. `angular_axis` is a
  required argument (not derived as "the other axis") since an N-D input has more than two axes to choose
  from. **Two `PFTImplementation` strategies**, dispatched like `DHTImplementation`/`DFTImplementation`:
  `HARMONIC_LOOP` (one `hankel_transform`/`inverse_hankel_transform` call per harmonic — a trailing batch
  axis rides along for free via those functions' own N-D support) and `STACKED_KERNEL` (every harmonic's
  kernel stacked into one `(n_angular, n_radial, n_radial)` array via `CachedBesselDHT._bessel_kernel`
  directly, applied with a single batched `numpy.matmul` across every harmonic and batch element at once).
  `DEFAULT_PFT_IMPLEMENTATION` is `STACKED_KERNEL`, picked by `benchmarks/run_pft_benchmarks.py`'s batched
  `(radial, angular, batch)` scenario (~4.7ms vs. `HARMONIC_LOOP`'s ~5.2-6.2ms at `n_angular=31,
  n_radial=128, batch=64`) — `HARMONIC_LOOP` stays faster on a single unbatched 2-D call (~2.7ms vs.
  ~3.6ms), but batching is the scenario this default is chosen for. The kernel stack itself is cached (an
  `lru_cache` keyed on the hashable `(grid, direction)`, `STACKED_KERNEL_CACHE_MAXSIZE`-bounded, mirroring
  `CachedBesselDHT`'s own `(n, size)` cache) — see `DESIGN_NOTES.md`, "PFT: `STACKED_KERNEL`'s kernel-stack
  cache is required for its own performance win," for why this cache is load-bearing.
- **Negative orders reuse the positive-order kernel.** `Y^{(-n)N} = (-1)^n Y^{nN}` exactly (the
  denominator's squared Bessel term is unchanged because `J_{n-1}(j_nk) = -J_{n+1}(j_nk)` at a zero of
  `J_n`), so `scaled_hankel` always calls the DHT with `abs(n)` and multiplies the sign in afterwards; the
  DHT's own `n >= 0` contract (`IntValidator.value_is_non_negative`) is untouched.
- **Per-harmonic scale factors**, on top of `hankel_transform`/`inverse_hankel_transform`'s own
  `R**2/j_nN`/`j_nN/R**2`: forward is `2*pi * i**(-n)`, inverse is `i**n / (2*pi)` (Appendix A-5/A-6).
- **Verified against the paper's own published figures, exactly**, at `N2=15, N1=383, R=40`: forward
  `E_avg`/`E_max` = −63.80/−8.38 dB, inverse `E_avg`/`E_max` = −98.03/−12.26 dB (measured to the paper's own
  precision by `tests/test_transform.py`). **`E_max` legitimately occurs at the grid's central gap and can
  turn positive at an inadequate grid size — every accuracy gate checks the average dB error, never the
  max.** Round-trip is a *regression* test only: the DHT's self-inverse kernel and the forward/inverse
  scale factors cancel exactly across a round trip regardless of whether the forward transform is itself
  numerically accurate at a given grid size, so round-trip can never certify accuracy.
- `tests/fixtures.py` holds the shared Gaussian oracle (`gaussian_f`/`gaussian_F`) and a code-generated
  Shepp-Logan phantom (`shepp_logan_phantom`, no binary test asset) used for a qualitative round-trip check,
  since the Gaussian oracle alone is circularly symmetric and would not catch every axis mix-up.

## Architecture: the ring-consistent route (`src/pypft/ring.py`)

The **exact path** (`forward_pft`/`inverse_pft`, the domain chain) stays the default everywhere — notebooks'
main line, quickstart, fixtures. `pypft.ring` is the clearly labeled alternative for accuracy against the
continuous transform: it avoids the spoke-to-harmonic identification on both sides, with no new DHT/DFT
kernel. Rationale, measurements and the trade-off: `DESIGN_NOTES.md`, "PFT: the exact path and the
ring-consistent route." Key points:

- **Three stages.** (1) *True rings in*: `sample_harmonics_cartesian(image, grid, *, n_quadrature=None)`
  (bilinear `cv2.remap` at `n_quadrature` angles per ring, `sample_cartesian`'s convention; default
  `max(n_angular, ceil(2*pi*R))`, `R` in pixels) or `sample_harmonics_uniform_polar(values, grid, *,
  radius)` (radial `CubicSpline` onto each order's `grid.r` row, quadrature over the data's own spokes —
  no angular interpolation) return a spatial-harmonic array on `angular_dft`'s scale (×`n_angular`), the
  values of a `PolarSpatialHarmonicSignal`. (2) the existing `scaled_hankel`. (3) `evaluate_frequency`
  evaluates each `F_n` at every spoke's own `rho_qm` through the DHT's Fourier-Bessel expansion and sums
  `(1/N2) sum_n F_n e^{i n psi_q}`; `evaluate_space` is the space-side mirror. `forward_pft_ring(values,
  grid, *, radius=None)`: `radius` omitted → Cartesian image, given → uniform polar data.
- **`_off_grid_kernel(order_in, zeros_out, zeros_in, output: Direction)`** is `(2/j_nN) J_n(u j_nk) /
  J_{n+1}(j_nk)**2` applied to `Y^{nN} x`, with `u = j_qm / j_nN` on the frequency side and `u = j_qm /
  j_qN` on the space side — the two sides differ only in that denominator (an easy bug to reintroduce:
  they are **not** the same matrix). At `q == n` both reduce to `Y^{nN}`. Harmonics `±n` and spokes `±q`
  share orders (`_orders(grid)`), so `(N2/2+1)**2` matrices of `N1**2` direct `jv` values are built per
  call.
- **Cost, thread pool, no cache.** The output stage dominates (~0.6s at 382x15, ~8s at 200x63, ~19s at
  576x48 on 16 cores). `scipy.special.jv` releases the GIL, so output orders run on a
  `ThreadPoolExecutor` (~10x); nothing is cached, since a per-grid cache would be 1.7 GB at 576x48. It is a
  single implementation, not a `PFTImplementation` strategy (a different discretization, not the same one
  faster).
- **`inverse_pft_ring(F, grid, *, radius=None)` with `radius` reads uniform polar frequency data** (k-space):
  `_sample_harmonics_uniform_polar(..., direction=Direction.INVERSE)` splines every data spoke onto each
  order's `grid.rho` row and sums over the data's own spokes -- the frequency mirror of the forward
  uniform sampler, with no central-gap extrapolation since uniform data starts at frequency 0. Its batch
  axis goes through `transform._validate_batch_axis` (factored out of `_validate_pft_input`). Measured:
  8.2e-8 (centered Gaussian) where the exact path is 0.34, and on measured cardiac k-space it is the only
  path giving a recognizable image (`DESIGN_NOTES.md`, "PFT: the exact path and the ring-consistent
  route").
- **`inverse_pft_ring(F, grid)` from grid samples is approximate**, on the grid's frequency spokes: per-spoke
  cubic spline onto every order's `rho_nm` (0 past a spoke's last sample; extrapolated inside the
  central gap), quadrature, inverse `scaled_hankel`, `evaluate_space`. It is accurate only when the
  frequency grid's central gap holds little of the transform (round trip 1.4e-4 at 382x15, R=40) and
  fails otherwise (1.6 at 200x63, R=10) — documented, not a bug to "fix" cheaply (zero/constant gap fills
  measured worse).
- `LimitKind.SPACE_LIMITED` only (`NotImplementedError` otherwise); trailing batch axis everywhere; the
  evaluators/`inverse_pft_ring` reuse `transform._validate_pft_input`. Tests: `tests/test_ring.py` (every
  tolerance a measured value with margin; ~5s).

## Architecture: typed domain objects (`src/pypft/domains.py`)

`PolarDomain(Enum)` names the four points a polar array occupies across the PFT/IPFT chain
(`SPATIAL_ANGULAR`/`SPATIAL_HARMONIC`/`FREQUENCY_HARMONIC`/`FREQUENCY_ANGULAR`). **Naming rule: one
domain enum per coordinate system, members name every coordinate group's state, no system prefix** —
first word the radial state (`SPATIAL`/`FREQUENCY`, changed only by the DHT), second the angular state
(`ANGULAR` physical angle / `HARMONIC` harmonic order, changed only by the angular DFT/IDFT). A future
`SphericalDomain`/`CylindricalDomain` gets its own enum by the same rule, without touching `PolarDomain`;
no shared abstract `Domain` base exists until a second system does. Rationale: `DESIGN_NOTES.md`,
"Domains: one state word per coordinate group."
`BaseSignal` (frozen dataclass: `values`, `grid`, `domain: ClassVar[PolarDomain]`, `batch_axis`) and its four
subclasses (`PolarSpatialAngularSignal`, `PolarSpatialHarmonicSignal`, `PolarFrequencyHarmonicSignal`,
`PolarFrequencyAngularSignal`) are a thin, optional wrapper over `pypft.transform`'s already-verified numerics —
`values`-in/`values`-out through `forward_pft`/`inverse_pft`/`scaled_hankel`/`angular_dft` remains the
primitive the numeric path never requires this module for. Key points:

- **One ordered `_POLAR_CHAIN` tuple, no `_LEGAL_MOVES` table.** The chain is a path graph (no branches, no
  cycles), so a transition is legal exactly when it moves one step along `_POLAR_CHAIN` — encoding legality
  separately would just duplicate it. Each subclass defines only the step methods for its own neighbours
  (named after the *destination* domain in full: `to_spatial_angular`/`to_spatial_harmonic`/
  `to_frequency_harmonic`/`to_frequency_angular`, so
  `signal.to_spatial_harmonic().to_frequency_harmonic().to_frequency_angular()` reads as the
  chain itself), so calling a step method that does not exist on a given subclass is a `pyright` error on a
  hand-written chain, not just a runtime `AttributeError`. `tests/test_domains.py` pins both the `_POLAR_CHAIN`
  order (by member name) and that every `_STEP_TOWARD`/`_STEP_BACKWARD` entry is `to_<destination>`.
  There are no deprecated aliases for any other names. `to(domain)` is the dynamic counterpart: a 5-line walk along `_POLAR_CHAIN`, never a general
  graph search, since the only decision at each step is which direction and which of `_STEP_TOWARD`/
  `_STEP_BACKWARD` advances one edge that way.
- **`batch_axis: int = DEFAULT_BATCH_AXIS`** mirrors `forward_pft`/`inverse_pft`'s own parameter exactly —
  `BaseSignal.__post_init__` reuses `pypft.transform._validate_pft_input` directly (the same
  sample-or-batch, batch-axis-after-the-sample-axes validation) rather than duplicating it, and every step method threads
  `self.batch_axis` through to the signal it returns, since batching never changes along the chain.

## Architecture: experimental acquisitions (`src/pypft/acquisition/`)

Converts measured radial k-space into what PyPFT reads. Three layers; rationale and the dataset
measurements in `DESIGN_NOTES.md`, "Acquisition: radial lines become pairs of spokes, at uniform angles
only," "Acquisition: a layout describes an array, never a file" and "Acquisition: the MD-CNN dataset's
line angles are measured."

- **Layout-independent** (`_kspace.py`): `radial_lines_to_polar(lines, *, angles, center_index)` takes
  `(readout, line[, batch])` lines through the k-space center and splits each at `center_index` into two
  spokes (forward half at the line's angle, backward half reversed at angle + pi), keeping
  `min(center + 1, readout - center)` samples. Angles must be multiples of `pi / L`, each once modulo pi,
  in any order; anything else (golden angle) raises `NonUniformAnglesError(ValueError)`. Returns
  `RadialKSpace(values, radius)` (frozen, `eq=False`; `radius = n_radial * 2 pi / readout` radians per
  pixel of the readout's inverse DFT), with `save`/`load` to `.npz` (`allow_pickle=False`).
- **Readers** (`_base.py`): `BaseRadialReader`, classmethod-only like the DHT strategies, with four hooks
  on **arrays, never files** (`_axis_sizes(shape)`, `_select_lines(values, *, index)`, `_line_angles`,
  `_center_index`) and template methods `selectable_axes(values)`/`read(values, *, selection)`. A
  *selection* maps each selectable axis to an index or `None` (keep whole); omitted axes take index 0;
  kept axes are flattened in C order into the one batch axis. The selection is applied in
  `_select_lines`, before the real/imaginary parts are joined. `RadialLineLayoutError(ValueError)` for
  arrays in the wrong layout.
- **Layouts**: `RadialLineLayout` enum + `_READERS` dict in `__init__.py` (`DHTImplementation`'s pattern);
  public `selectable_axes(values, *, layout)` and `select_radial_kspace(values, *, layout, selection)`.
  **Naming rule: a member names the array's axes in order, never a dataset, scanner, file format or
  "acquisition"** -- the same layout reads measured and generated data, from any file, alike; provenance
  goes in docstrings. One layout so far, `SLICE_PHASE_READOUT_LINE_COIL_COMPLEX`
  (`SlicePhaseReadoutLineCoilComplexReader` in `_slice_phase_readout_line_coil_complex.py`): a real
  `(slices, phases, readout, lines, coils, 2)` array (complex raises), selectable `slice`/`phase`/`coil`,
  line `l` at `l * pi / L` and center `readout // 2`. The MD-CNN dataset is stored in it (MATLAB variable
  `data`); its angles/center are measured, not documented by the dataset, and its orientation (mirror) is
  undetermined.
- **Files** (`_files.py`, layout-free): `ARRAY_FILE_SUFFIXES` (`.npy`, `.npz`, `.mat`);
  `holds_several_arrays(path)` (suffix only: `.npz`/`.mat`); `stored_arrays(path)` lists a multi-array
  file's numeric arrays and shapes from headers only (`np.lib.format` header of each `.npz` member,
  `scipy.io.whosmat`; structs/cells/text dropped; `ValueError` for `.npy` or none numeric);
  `load_stored_array(path, *, name=None)` (`name` required for `.npz`/`.mat`, forbidden for `.npy`;
  unreadable files, MATLAB 7.3 included, and non-numeric arrays become `ValueError`).
- **`app.py`**: `AcquisitionApp` (Textual, one screen: file, layout, Inspect, an array `Select` shown only
  for a multi-array file -- filled on the first Inspect of a file, its choice read on the next -- one
  field per axis taking an index or `all`, output `.npz`, Overwrite checkbox, Convert, Quit; Convert
  re-reads the inspected file and array; errors shown on the status line, never raised) and `main(argv)`; `__main__.py` runs it. `AcquisitionApp.status` exposes the status text so tests
  need no Textual internals. `textual` is a runtime dependency with one floor, `1.0.0` (the oldest
  `textual-fspicker` supports; `tests/acquisition`/`tests/cli` pass with it on 3.12, 3.13, 3.14), chosen
  over a GUI because it needs no Tkinter. `textual-fspicker>=0.6.0` (path completions, 3.14) gives
  `pypft.cli` its file dialog; `rich>=13.4.2` (the first not capping markdown-it-py below the docs
  toolchain's 4.x) is imported for `rich.text.Text`: textual 1.0's `Static.update(str)` parses markup
  even with `markup=False`, so both apps' status lines update with a `Text`. Tests drive it with `run_test()` inside `asyncio.run` (no pytest plugin)
  and press buttons via `Button.press()` then poll `app.status`: a mouse click is ignored while the button
  still shows its previous click, and `pilot.pause()` can return before an async handler finishes mounting.
- Tests: `tests/acquisition/` (`test_kspace.py` analytic k-space, `test_readers.py` on the fixture and
  synthetic arrays, `test_files.py` the file layer -- the same array from all three formats converts
  alike -- `test_app.py`). `np.testing.assert_array_equal` takes its arrays positionally
  there: its keyword names differ between the numpy floor (`x`/`y`) and numpy 2.4 (`actual`/`desired`).

## Architecture: test samples (`src/pypft/samples/`)

Measured samples too large for the repository and the wheel are downloaded, never committed. `Sample`
(an `Enum` whose value is the output file name) maps through the private `_RECIPES` dict to a `_Recipe`:
a `RemoteFile(url, filename, md5)` source plus `make(source, output)`, which writes the sample from the
downloaded source. Adding a sample = a `Sample` member, its recipe (its cutting code in its own private
module, like `_md_cnn.py`), and a `.gitignore` line; `tests/test_samples.py::test_every_sample_has_a_recipe`
pins the registry's completeness.

- `download_file(remote, *, directory)` streams to `<name>.part`, checks the publisher's MD5
  (`ChecksumError(ValueError)` otherwise) and renames only on a match; a cached file with the right MD5 is
  reused. It sends `User-Agent: pypft` (`USER_AGENT`): Harvard Dataverse answers urllib's default with 403.
- `fetch_sample(sample, *, directory, cache_directory=None, overwrite=False)` returns an existing sample
  untouched unless `overwrite`; without `cache_directory` the source goes to a `TemporaryDirectory` and is
  deleted. The sample is written as `<stem>.part<suffix>` (scipy's `savemat` appends `.mat` to other
  names) and renamed when complete.
- `MD_CNN_EXCERPT` (`_md_cnn.py`): Dataverse file `doi:10.7910/DVN/CI3WB6/XS8ZVS`
  (`P01_slc1_phs25_lins196_cols832_cha16.mat`, 233 MB, MD5 `e1a7ff0a...`), cut by `make_excerpt` to the
  first slice and phase, the central 416 readout samples (center at 208), the four strongest coils (14, 5,
  10, 13 sorted), float32, compressed (1.9 MB), in the dataset's own 6-D layout.
- Tests never touch the network: they download local files through `file://` URLs and monkeypatch
  `_RECIPES` to a synthetic dataset file. Tests for this package live in `tests/test_samples.py`, not
  `tests/samples/`, which is the fixture data directory.

## Architecture: the terminal interface (`src/pypft/cli/`)

`python -m pypft` / `pypft` opens `PyPFTApp` (Textual). It adds no numerics; rationale for the event-loop
and warning handling: `DESIGN_NOTES.md`, "Terminal interface: the app owns the event loop and the screen."
Three private modules, layered so only `_app.py` touches Textual:

- **`_io.py`** (UI-free): `load_array(path, *, batched) -> LoadedInput(values, radius, is_image)` reads
  `.npy` (any array), `.npz` (`RadialKSpace.save` output; carries its radius, IPFT input only) and
  grayscale `.tif`/`.tiff`/`.png` (one real spatial sample); empty/corrupt files become `ValueError`
  (`np.load` leaks a handle on a broken archive, hence the own `open` and the `zipfile.is_zipfile` check).
  `POLAR_FILE_SUFFIXES` are `load_array`'s, `LINE_FILE_SUFFIXES` (= `pypft.acquisition.ARRAY_FILE_SUFFIXES`)
  radial lines', `SUPPORTED_SUFFIXES` their union; `load_array` refuses `.mat` ("read only as radial
  k-space lines"). `acquisition_input(values, *, layout, selection)` converts an already loaded array of
  lines via `select_radial_kspace` (single sample or batch decided by the selection).
  `output_filename` -> `<stem>_<domain>_<magnitude|phase>[_b<NNN>].png` (domain slug is
  `viz._domain_label`, index width `max(3, digits of batch_size - 1)`); `save_signal_figures` renders on
  bare `Figure` objects (never pyplot) and raises `FileExistsError` unless `overwrite`;
  `warn_if_slow` emits `CLIPerformanceWarning` above `SLOW_BATCH_SIZE` (32); `record_warnings(call)`
  runs a call under `catch_warnings(record=True)` + `"always"` and returns the messages as text.
- **`_pipeline.py`** (UI-free): `InputGrid` (`ON_GRID`/`UNIFORM_POLAR`), `Route` (`EXACT`/`ACCURATE`),
  `GridSettings` (radius required exactly for uniform input), `check_grid` (adequacy/Nyquist warnings as
  text), `suggested_grid(shape, radius)` for k-space (`R = n * pi / radius`, a quarter of the spokes,
  the largest `n_radial` with `max(grid.rho) <= radius` via `j_(m,k) < (k + m/2 - 1/4) pi`; 183x98,
  R=208 on the MD-CNN excerpt), `run_transform(loaded, settings, *, direction, route, trace)`. Exact path: uniform input goes
  through `resample_uniform_polar(_frequency)` first (with an interpolation notice); `trace` uses
  `forward_pft_traced`/`inverse_pft_traced`. Accurate path: `forward_pft_ring(radius=...)` (uniform input
  only) / `inverse_pft_ring`. Rejected: an image for the IPFT, `.npz` for the PFT or as on-grid, trace
  off the exact path, on-grid shape mismatch.
- **`_app.py`**: the layout `Select` offers `SampleLayout` members plus every `RadialLineLayout`
  member (`layout_options()`, `InputLayout = SampleLayout | RadialLineLayout`, stored in
  `Session.layout`); with a radial-line layout chosen, File lists a `.npz`/`.mat` file's arrays
  (`stored_arrays`) -> `ArrayScreen` (an `#array` `Select`, labels from `pypft.acquisition.app.array_label`),
  or reads a `.npy` file's array directly; `acquisition_screen(path, layout, name)` loads the array and
  runs `selectable_axes` so a wrong array is reported on the screen before -> `AcquisitionScreen` (holds
  the array; one `axis-<name>` `Input` per axis, starting at `0`, taking an index or `all` via
  `pypft.acquisition.app.parse_axis_choice`) -> Grid, whose fields k-space prefills from
  `suggested_grid`. The file picker, `FilePickerScreen`, subclasses `textual_fspicker.FileOpen`:
  its listing is filtered to `input_suffixes(layout)` (`is_input_file(path, suffixes)`, any case), and
  its field takes a pasted path -- a directory opens in the dialog, a file is picked only if
  `_should_return` accepts it (exists, accepted suffix; errors show as the dialog's border subtitle). The layout screen's `#layout-note` shows
  `layout_note(layout)` (`_RADIAL_LINE_LAYOUT_NOTES`, one text per `RadialLineLayout` member) and is
  hidden for polar layouts; textual 0.39 posts `Select.Changed` only on a user pick, so tests post it. `StepScreen` base (Back except on Start, Quit, a status line; `act(button_id)` errors in
  `_REPORTED_ERRORS` are shown, never raised) and Start -> Tutorials | Sample -> File (+ `FilePickerScreen`
  dialog) -> Grid -> Direction -> Result (+ `ConfirmScreen` before
  overwriting). On-grid input takes its grid size from the array; uniform input's `n_radial`/`n_angular`
  are editable (measured k-space cannot be covered by a grid of its own shape). Transform and save
  report `"...`-suffixed status first and run via `call_after_refresh`. View draws into **one** figure
  (redrawn per batch element): a pyplot window pumped by `set_interval` + `flush_events` when
  `interactive_backend()`, else a bare `Figure` saved to a temp file; the window figure is closed in
  `ResultScreen.on_unmount`. `TUTORIALS_URL` is the docs' tutorials page; the JupyterLab button appears
  only for a clone with `jupyterlab` installed.

Tests: `tests/cli/` (`test_io.py`, `test_pipeline.py`, `test_app.py`), driven like `tests/acquisition`
(`run_test()` inside `asyncio.run`, `Button.press()`, then waiting for the screen or a non-`...` status
to change). The window path is tested by monkeypatching `_app.interactive_backend` and `Figure.show`.

## Architecture: the explicit PFT kernel oracle (`src/pypft/_kernel.py`)

`kernel_matrix(grid, *, direction)` builds the PFT's combined `E^{-}`/`E^{+}` operator directly from Yao &
Baddour's own definition (PeerJ CS Part II, Eqs. 1-3) — one `(n_radial, n_radial)` DHT kernel per harmonic
(via `NaiveDHT._bessel_kernel`, not `pypft.dht`'s cached entry points, to stay independent of the code under
test) Kronecker-multiplied against that harmonic's `(n_angular, n_angular)` angular phase factor and summed.
Applying the resulting `(n_radial*n_angular, n_radial*n_angular)` matrix to a raveled input and reshaping
reproduces `forward_pft`/`inverse_pft` exactly (`tests/test_kernel.py`) — a **stronger** check than the
Gaussian oracle above, since it certifies the FFT-plus-per-harmonic-Hankel composition against the paper's
direct definition rather than only against a known analytic answer. Deliberately named with a leading
underscore (unlike `pypft.dft`, which is internal but still unprefixed): building this matrix costs
`O(n_angular * n_radial**2)` and touches all `O((n_radial*n_angular)**2)` of its entries, so it exists purely
as a from-scratch test oracle, never as a faster or more convenient public entry point. Sign convention
matches `scaled_hankel` exactly, including its own `_harmonic_sign` — duplicated here rather than imported,
so a bug shared between the two modules cannot cancel itself out in the comparison.

`tests/test_transform_properties.py` exercises the Mathematics 7(8):698 (Part I) operational rules this
kernel obeys — orthogonality (Eqs. 34, 37), the complex-exponential/delta pair (Eq. 43), the generalized
shift operator and its shift-modulation/modulation/convolution/multiplication consequences (Eqs. 46-71,
built from `kernel_matrix` columns exactly as `tests/dht/test_kernel_properties.py` builds them from the raw
DHT kernel — see below), and the non-symmetric-kernel generalized Parseval relationship (Eqs. 80, 87-88,
using a "starred" conjugate built from the *opposite*-direction kernel, since the plain conjugate only
preserves the inner product for the *symmetric* kernel choice this package does not implement anywhere).
Properties about the transform itself rather than the kernel alone — rotation equivariance (Eq. 75),
linearity, the harmonic-0/DC term, and a real signal's own harmonic spectrum having a twisted conjugate
symmetry (`F_{-n} = (-1)^n * conj(F_n)`, not plain conjugate symmetry, because of `scaled_hankel`'s own
negative-order sign) — are tested directly against `forward_pft`/`inverse_pft` instead, since that is both
simpler and closer to how a caller would actually observe them.

## Architecture: visualization (`src/pypft/viz.py`)

`Axes`/`Figure`-based figures for a `pypft.domains.BaseSignal`. `plot_signal`/`render_cartesian`
never write to disk, and `forward_pft_traced`/`inverse_pft_traced` don't either on their own —
only `PFTTrace.save` does, and only when a caller calls it explicitly. Key points:

- **`plot_signal(signal, ax=None)` renders every domain the same way — a gamma-enhanced magnitude
  and a phase map**, unconditionally, with no domain-based dispatch
  (`matplotlib.colors.PowerNorm`, never a hand-rolled `** gamma`, so the colorbar's own tick values stay
  meaningful). Every `PolarDomain` member is treated as complex-valued: `SPATIAL_HARMONIC` is an angular
  DFT's own coefficients, generically complex even when the space-domain signal being transformed is real
  (a DFT of real input is only symmetric, not real, in general); `SPATIAL_ANGULAR` can likewise carry a
  non-trivial phase after a full forward-then-inverse round trip. Since
  every domain takes the same path, `plot_signal` always returns `tuple[Axes, Axes]`, and there is no
  `_plot_complex_signal`/`_plot_magnitude_signal` helper split, since there is no dispatch for such a
  helper to support.
  `plot_signal` still type-validates a given `ax` as a 2-tuple *before* unpacking it into
  `(magnitude_ax, phase_ax)` — an unguarded unpack would raise `ValueError` instead of `TypeError` on a
  wrong-length `ax`, breaking `plot_signal`'s own documented type-validation contract. `BaseSignal.plot()`
  is a thin delegate to this function, using a **function-local** import of `pypft.viz` (`domains.py`
  cannot import it at module level: `pypft.viz` itself imports `PolarDomain`/`BaseSignal` from `pypft.domains`,
  so a top-level import would be circular).
- **`_magnitude_cmap(domain)` grayscales `SPATIAL_ANGULAR`'s own magnitude specifically**, everywhere a
  magnitude is drawn (`plot_signal`, `render_cartesian`) — a
  `SPATIAL_ANGULAR` magnitude is literally a photographic image (`pypft.grid.sample_cartesian`'s own image
  argument), unlike every other domain's more abstract Fourier/harmonic-coefficient magnitude, which stays
  at `matplotlib`'s own default colormap (`cmap=None`, resolved by each `imshow` call). Phase panels are
  never affected by domain — `cmap="twilight"` (a cyclic colormap, correct for any wrapped `[-pi, pi]`
  quantity regardless of domain) is unconditional, and so is its color *range*: `_PHASE_VMIN`/`_PHASE_VMAX`
  (`-np.pi`/`np.pi`) are always passed explicitly to every phase `imshow` call — see below for why.
- **Every phase `imshow` call pins `vmin`/`vmax` to `[-pi, pi]` explicitly**, rather than leaving the range
  to `matplotlib`'s own auto-scaling — see `DESIGN_NOTES.md`, "Visualization: phase color range is pinned
  to `[-pi, pi]`," for why (a degenerate-`Normalize` failure mode for any exactly-constant phase, e.g. a
  real-valued `SPATIAL_ANGULAR` signal). `tests/test_viz.py::test_plot_signal_phase_color_range_is_fixed_for_constant_phase`
  pins this down: `phase_ax.images[0].get_clim() == (-np.pi, np.pi)` must hold for a real-valued signal.
- **`render_cartesian(signal, *, height, width, ax=None)`** is the display-only counterpart for the two
  non-harmonic domains specifically (`SPATIAL_ANGULAR`/`FREQUENCY_ANGULAR`, `_PHYSICAL_ANGLE_DOMAINS` — the two
  whose angular axis is a physical angle, not a harmonic order): it interpolates `PolarGrid.r`/`.theta`'s own non-uniform sample points onto
  an ordinary Cartesian pixel grid via `scipy.interpolate.griddata`. This is an **approximation for display
  only** — its output must never be fed back into `forward_pft`/`inverse_pft`, unlike
  `pypft.grid.sample_cartesian`'s own exact, order-dependent sampling. **`grid.theta` must be negated before
  use here** (`y = -grid.r * np.sin(grid.theta)`, not a bare `sin`) — see `DESIGN_NOTES.md`,
  "Visualization: `render_cartesian`'s `grid.theta` must be negated," for why.
  `tests/test_viz.py::test_render_cartesian_orientation_matches_image_convention` pins this down: a bright
  wedge at `theta=+pi/2` must render in the physical lower half.
- **`forward_pft_traced`/`inverse_pft_traced`** walk the same `pypft.domains.BaseSignal` chain
  `forward_pft`/`inverse_pft` are built on (rather than duplicating the pipeline), so their `values` are
  identical to calling `forward_pft`/`inverse_pft` directly. They return a `PFTTrace` (frozen dataclass:
  `values`, `signals` — all 4 domains visited, in order — `figures`, and `figure_labels`), **always the same
  type regardless of** `visualize_steps`/`visualize_pipeline` — a value-dependent
  return type is a `pyright` defect, which is why tracing is a separate entry point rather than a
  `visualize=`/`record=` flag on `forward_pft`/`inverse_pft` themselves. `visualize_steps` builds one figure
  per domain (4), each rendered via `plot_signal`; `visualize_pipeline` builds one holistic mosaic figure
  (two panels per domain, via `plt.subplots` with a computed `ncols`, each panel likewise rendered via
  `plot_signal`). There is deliberately no option to render a traced
  panel as a Cartesian circle instead of the ordinary polar-index table, to keep this module's surface
  area proportional to its value: `render_cartesian` (above) remains the one function for rendering a
  polar signal as a Cartesian circle, called directly by a caller who wants one, e.g. on `trace.signals[0]`/
  `trace.signals[-1]`.
- **`PFTTrace.figures`/`figure_labels` and `filterwarnings = ["error"]`**: `figures: tuple[Figure, ...] =
  field(default=(), compare=False, repr=False)` (a `Figure` is neither comparable nor usefully
  representable) plus a `PFTTrace.close()` method are both required because matplotlib's own "more than 20
  figures have been opened" `RuntimeWarning` becomes a **test failure** under this project's
  `filterwarnings = ["error"]`. Tests must assert `plt.get_fignums()` actually returns to its prior length
  after `close()`, not just that the return types are right. `figure_labels` is a same-length, parallel
  tuple of names (`"step_<system>_<member>"`, e.g. `"step_polar_spatial_angular"`, the system taken from
  the enum's class name by `viz._domain_label`, and `"pipeline"`) `PFTTrace.__post_init__` validates against `figures`' own
  length — what `PFTTrace.save(directory)` names each `"<index>_<label>.png"` file after. `save` is the
  **one** explicit, opt-in place this module ever writes to disk: never called
  implicitly by tracing itself, and it creates `directory` (`Path.mkdir(parents=True, exist_ok=True)`) after
  validating it with the shared `PathValidator`.
- **Cross-module RST type references must be fully qualified.** A docstring's `:type:`/`:rtype:` field
  naming a type defined in *another* module (e.g. `pypft.domains.BaseSignal`, `pypft.grid.PolarGrid`) must
  spell out the dotted path, not the bare class name — `sphinx-build -W` raises a `more than one target
  found for cross-reference` (`ref.python`) warning otherwise, since both the bare name (re-exported at
  `pypft.<Name>`) and the fully-qualified original resolve as candidates. A module documenting its *own*
  type (e.g. `pypft.grid`'s own docstrings naming `PolarGrid`) stays bare, since Sphinx resolves an
  unqualified reference to the current module first. `transform.py`'s existing `:type grid:
  pypft.grid.PolarGrid` already followed this; `viz.py` is what makes the rule explicit.
- **`tests/conftest.py` forces the `Agg` backend** (`matplotlib.use("Agg")`) before any test imports
  `matplotlib.pyplot` — every CI OS is headless, so an interactive backend (e.g. `TkAgg`) fails outright.
  This only matters for plain `pytest` collection: a notebook executed via `nbmake` runs under a real
  Jupyter/`ipykernel` kernel, which usually defaults to the inline backend; every notebook still starts its
  first code cell with `%matplotlib inline` for viewers whose kernel defaults differently (enforced by
  `tests/test_notebook_style.py`).

## Notebook citations and `CITATION.cff`

There is no citation code in `src/`. A tutorial notebook cites a paper with a numbered in-text link,
`[[1]](#ref-1) (Eq. 41)` -- equation/figure/appendix locators always in parentheses *outside* the link,
never a bare "Eq. 21" -- to an entry of its own final Markdown cell, headed `## References`, whose entries
read `1. <a id="ref-1"></a> N. Baddour, "...," ... [doi:...](https://doi.org/...)`. Numbering is per
notebook, `1..K` in first-citation order. Nothing executes, so it renders unexecuted. Verified: links jump in
JupyterLab (which stores the anchor as `data-jupyter-id` and resolves `#ref-k` against it) and in the
Sphinx build; on GitHub (not checked live) the links may not jump, since GitHub rewrites `id`s. `tests/test_notebook_citations.py` checks, per notebook: every `](#ref-k)` link has its anchor,
every entry is cited, contiguous first-citation numbering (list number == anchor number == link text),
the References cell is the final cell of a citing notebook (and absent otherwise), every locator is paired
with a citation, no author-year `[Baddour2019a]` labels or `bibliography(`/`pypft.references` remain, and
every entry's DOI is listed under `references:` in the root `CITATION.cff`.

`CITATION.cff` (CFF 1.2.0, validated with `uvx cffconvert --validate`) is how PyPFT itself is cited
(GitHub's "Cite this repository" button) and is the canonical list of cited papers: every notebook
References entry's DOI must appear under its `references:` (the entry text itself is copied, not checked); README's "Citing" section points at it. Author is `alias: GhDamghani`, matching `pyproject.toml`.
MyST cannot see raw-HTML anchors, so `docs/conf.py` sets `nitpick_ignore_regex = [("myst", r"ref-\d+")]`
to silence exactly those `myst.xref_missing` warnings under `-W` -- see `DESIGN_NOTES.md`, "Notebooks:
citations link to raw-HTML anchors in a References cell."

## Architecture: the discrete Hankel transform (`src/pypft/dht/`)

Implements the DHT as a transform in its own right (not a discretized integral), following N. Baddour,
"The Discrete Hankel Transform" (2019, `.local_files/sources/baddour2019.md`) — read that source before
changing any of the kernel math. Key points:

- **Kernel choice**: the transform uses Baddour's `Y^{nN}` formulation (paper's Eq. 39), not the
  alternative symmetric `T^{nN}` formulation (Eq. 44) — see `DESIGN_NOTES.md`, "DHT: kernel formulation is
  `Y^{nN}`, not the symmetric `T^{nN}`," for why; `tests/dht/test_gaussian.py` and
  `src/pypft/dht/_base.py`'s module docstring carry the same rationale.
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
- **The kernel's Bessel values are always computed via a direct `jv()` call per order** — see
  `DESIGN_NOTES.md`, "DHT: the kernel's Bessel values must be computed directly, never via order
  recurrence," for why, including `DHT_ORDERS`' high-order coverage (16/32/64) and `dht_tolerance`'s
  order-dependent model.
- **Order-dependent tolerance:** even the correct kernel's self-inverse residual grows with order (an
  inherent discretization effect, not a bug) — from ~1.9e-9 at order 0 to ~1.1e-5 at order 64 (size 64). A
  flat tolerance is either too loose (hiding a regression) or too tight (rejecting the correct kernel above
  ~order 24), so order-sensitive assertions in `tests/dht/` use `tests/dht/tolerance.py`'s
  `dht_tolerance(order, size)` model instead of the flat `RTOL`/`ATOL` in `tests/dht/conftest.py`.
- **Negative-order relation and the shift/modulation/multiplication/convolution rules** (baddour2019.md's
  Transform Rules section) are exercised in `tests/dht/test_kernel_properties.py`. `Y^{(-n)N} = (-1)^n *
  Y^{nN}` is verified against a locally-defined `_general_bessel_kernel` (built for any integer `n`, unlike
  `BaseDHT._bessel_kernel`'s `n >= 0`-only contract) — neither `baddour2019.md` nor the Part I paper states
  this identity, so it is not attributed to either source; it follows independently from
  `J_{-n}(x) = (-1)^n * J_n(x)` (an exact Bessel identity for every `x`, not only at a zero). The four
  transform rules are all built from one `_generalized_shift(kernel, transform, k0) = kernel @
  (kernel[:, k0] * transform)` helper (Eq. 70: the shift of a signal is defined via *its own transform*,
  since the ordinary `f_{k-k0}` shift is unusable — the index can fall outside `[1, N-1]` and, unlike the
  DFT's exponential, the Bessel kernel does not wrap around); self-inverse `Y` makes the same helper
  direction-agnostic, serving both the shift-modulation (Eq. 76-77) and modulation-shift (Eq. 82) rules, and
  the convolution (Eq. 83-86) and multiplication (Eq. 89-92) rules built on top of it.
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

## Architecture: the angular discrete Fourier transform (`src/pypft/dft/`)

The centered angular DFT/IDFT that sits between PyPFT's stored, centered-angular arrays and the DHT's
per-harmonic processing — internal plumbing (no notebook; see the Notebooks section), exercised end-to-end
once `transform.py`'s PFT pipeline exists, and directly by `tests/dft/test_oracle.py` in the meantime. Same
strategy shape as `pypft.dht`:

- **Strategy pattern**: `src/pypft/dft/_base.py`'s `BaseDFT` defines two overridable hooks — `_forward`/
  `_inverse` — that call the raw, natural-order FFT/IFFT with no opinion on centering. The public
  `forward`/`inverse` template methods (not overridden) own the centered-angular convention themselves:
  they reorder to natural order, delegate to the hook, then reorder back to centered order, via
  `pypft.axes._center_angular`/`_uncenter_angular` — exactly the DHT's `_bessel_kernel`/`_apply` split,
  renamed for the DFT's single optimization axis (which FFT library computes the transform).
- **Implementations**: `_numpy.NumpyDFT` (`numpy.fft.fft`/`ifft`) and `_scipy.ScipyDFT` (`scipy.fft.fft`/
  `ifft`). `ScipyDFT`'s hooks return `cast(np.ndarray, ...)` — `scipy.fft`'s backend-dispatch decorator
  otherwise makes `pyright` infer a dispatch-machinery return type instead of the actual array.
- **Selection**: `DFTImplementation` maps to these two classes via `_IMPLEMENTATIONS` in
  `src/pypft/dft/__init__.py`. `DEFAULT_IMPLEMENTATION` is `NUMPY`, picked by
  `benchmarks/run_dft_benchmarks.py`'s repeated-forward-call scenario (essentially tied with `SCIPY` there,
  ~11.7us vs. ~12.1us). See `DESIGN_NOTES.md`, "DFT: `SCIPY_WORKERS` is not offered as a separate
  strategy," for why there is no `workers=-1` implementation, and the constant's own docstring and the
  benchmark report for the full numbers.
- **`harmonics(n_angular)`/`AngularParity`**: also live in `src/pypft/dft/__init__.py`, since this module
  owns the harmonic-range derivation. `harmonics` returns `-(n_angular // 2) .. n_angular - n_angular // 2
  - 1`, correct for either parity — there is deliberately **no `value_is_odd` validator**: an even angular
  sample count is fully valid, just with one asymmetry (below).
- **The even-`N2` Nyquist caveat**: for even `n_angular`, harmonic `-n_angular // 2` has no
  `+n_angular // 2` partner in `harmonics`' range, so a real-valued signal's usual conjugate symmetry
  (`X[-n] == conj(X[n])`) is one-sided at that one bin. Both `tests/dft/test_parity.py` (a direct
  conjugate-symmetry check) and `tests/dft/test_oracle.py` (reproducing the published PFT error figures at
  `N2=15/16/17`, composing this module with the existing DHT) exist specifically to keep this regression
  visible without waiting for the full PFT pipeline.

## Documentation (`docs/`)

A Sphinx skeleton (`furo` theme, `myst_nb` for notebooks, plain `autodoc` for the API reference — docstrings
are RST, not Google/NumPy style, so no napoleon extension is needed). Built with `sphinx-build -W docs
docs/_build` (warnings fail the build), part of the quality gate.

`notebooks/` is tracked at the repo root, not under `docs/`, so it can be shared as-is with `nbmake` (see
below). Sphinx requires toctree documents to live under its own source directory, so `docs/conf.py`
registers a `config-inited` hook that copies `notebooks/*.ipynb` into `docs/_notebooks/` before Sphinx reads
its sources; `docs/tutorials.rst`'s toctree globs `_notebooks/*`. This avoids a symlink (which needs
elevated privileges on Windows, one of CI's three platforms). `nb_execution_mode = "off"` — notebooks are
executed and checked by `nbmake` in CI, not re-executed by the docs build. `myst_enable_extensions =
["dollarmath"]` is what makes notebook `$...$`/`$$...$$` math render in the docs at all (without it the
dollars show literally, with no build warning). Display math inside a list item needs a blank line and a
two-space indent before its `$$...$$` line, or the item's text and the math merge into one paragraph.

`docs/conf.py`'s `exclude_patterns` must list `"jupyter_execute"` alongside `"_build"`: even with
`nb_execution_mode = "off"`, `myst_nb`'s jupyter-cache machinery writes a `docs/jupyter_execute/` directory
directly under the *source* tree (not under `docs/_build/`) as a side effect of a second `sphinx-build -W`
run against an already-pickled environment. Without the exclude, Sphinx then discovers those `.ipynb`
files as new source documents outside any toctree on the next build and `-W` fails on the resulting
warning — reproducible with zero other changes, purely by running `sphinx-build -W docs docs/_build` twice
in a row. `docs/jupyter_execute/` is gitignored either way, so deleting it (along with `docs/_build/`,
`docs/_notebooks/`, `docs/.jupyter_cache/`) before a build is always safe if it ever reappears.

## Notebooks

`notebooks/` is tracked and executed in CI via `nbmake` (see the CI section above), as one incremental
tutorial sequence where each notebook assumes only its predecessors (`docs/tutorials.rst` globs
`_notebooks/*`, so the numeric prefix is the rendered order):

- `00_installation_and_quickstart.ipynb` — math-first description of the PFT (rendered pipeline equation,
  radial MRI named only as an application), a `pip`-first install section (venv on macOS/Linux/PowerShell/
  cmd, `py -3.12 -m venv`, conda, `python -m pip`, editable install from a clone, one line for uv), and a
  commented end-to-end tour: `PolarGrid(n_radial=1024, n_angular=31, R=120)` (silent under
  `check_adequacy`) → `sample_cartesian(...).T` (with a one-line pointer to `resample_uniform_polar` for
  uniform polar data) → `forward_pft`/`inverse_pft` → `render_cartesian`/`plot_signal`, ending with a linked list of every later notebook.
- `01_polar_images_and_grids.ipynb` — sampling grids in general (Cartesian vs. uniform polar figure),
  PyPFT's angle convention stated directly (`atan2(y, x)` on image coordinates, `+x` toward `+y`, i.e.
  clockwise on screen), `cartesian_to_polar`/`polar_to_cartesian` round trip with an error map, the
  angular-origin check visualized (wedge, polar image and angular profile with measured/predicted
  `axvline`s, then an `assert`), the three-grid framing (uniform grid = your data, `PolarGrid` = the
  transform's grid, per-spoke interpolation = the bridge), and a `PolarGrid`/`sample_cartesian`/`plot_signal`/
  `render_cartesian` preview (pointing at `resample_uniform_polar` for uniform arrays) with a same-spokes-different-radii overlay figure. `polar_to_cartesian` returns exactly `0` outside the inscribed circle (its inverse
  `warpPolar` call passes `WARP_FILL_OUTLIERS`; the default transparent border leaves unreachable
  pixels as uninitialized memory), so the notebook needs no masking of its own.
- `02_sampling_grids.ipynb` — `PolarGrid`: why `grid.r` is 2-D (one row per spoke, order `abs(p)`,
  the spoke-harmonic identification cited to Part I Eq. 18 / App. A.6), a row-equality matrix (only ±n rows coincide: Bourget/Siegel) and per-order spacing,
  the paper's scatter figures (clockwise `set_theta_direction(-1)`), a polar-pixel figure (annular-sector
  cells colored by area over a Cartesian pixel, `LogNorm` centered on 1, `y` down), the central gap (gap
  radius per harmonic and vs. `N1`, forward pointer to `05` for `E_max`), the even-`n_angular` Nyquist
  bin, `check_adequacy`, and `check_nyquist_adequacy` with Part II's Eqs. 20/21 rendered and a
  zeros-vs-order plot (`band_limit` 30 passes, 31 warns at `N1=384, R=40`), then "Uniform polar
  samples and the transform's grid": the three-part answer, radius of one index across spokes, spurious
  harmonics bar plot, the same uniform array fed as-is vs. through `resample_uniform_polar` (relative L2),
  and a 63-spoke `E_avg`-hides-it case.
- `03_discrete_hankel_transform.ipynb` — continuous Hankel transform in the package's convention
  (`J_n(rho r)`, no `2*pi`; `rho` in radians per length unit, `rho_nk = j_nk / R`), its 2-D-FT origin
  (`2*pi * i**(-n)`), `R` and the sample points, the self-reciprocal `exp(-r^2/2)` Gaussian, the `Y^{nN}`
  matrix built in-notebook by a `dht_kernel(order, size)` helper (any integer order), self-inverse
  (approximate per Baddour's Eq. 37), delta pair, negative orders, the generalized shift (order-0 bump
  shifted to rings at `r_{0k0}`) and its four rules checked with the dimensionless `Y` transform.
- `04_domains.ipynb` — the four domains as the map of the chain, an off-center Gaussian walked
  `to_spatial_harmonic().to_frequency_harmonic().to_frequency_angular()` with every
  intermediate drawn by `signal.plot(ax=...)`, illegal-move `AttributeError`, a step back, `to(domain)`;
  `forward_pft` is only pointed forward to.
- `05_pft_and_ipft.ipynb` — `forward_pft`/`inverse_pft` at Yao & Baddour's `N2=15, N1=383, R=40`,
  `plot_signal` + `render_cartesian` of `f`/`F`, dB error map (index table, frequency plane near the
  center, error vs. `rho`, histogram), the inverse figures, and an inadequate-grids section: `(382, 63)`
  fails `check_adequacy`, `(96, 15)` fails both checks (`band_limit=12`), warnings recorded with
  `warnings.catch_warnings(record=True)`; forward `E_avg` degrades while the round trip stays at rounding
  error; relative L2 reported beside every `E_avg`; an off-center-Gaussian section showing where the
  error sits; then "Two paths": exact path vs. `forward_pft_ring` from uniform samples on the same
  off-center Gaussian (error maps, relative L2, round trip), when to use which, the maze both ways
  (`maze_polar.tif` vs. `sample_maze_rings`, imported from `../scripts/make_maze.py` via `importlib`,
  ~40s), and an analytic disk check on a 192x48, `R=1` grid (exact path 2.70, ring 0.067). `02`'s
  uniform-grid section points to it.
- `06_batches.ipynb` — "Batches of samples": building a batch, transforming it, the
  batch-axis-after-the-sample-axes `ValueError`, `np.moveaxis`, `BaseSignal.batch_axis`, plotting one
  sample at a time with `render_cartesian`. No timing or
  "exactness" sections.
- `07_pft_properties.ipynb` — builds `E^-`/`E^+` in-notebook from Part I's Eq. 32 (`pft_kernels`,
  dimensionless; `forward_pft = 2*pi*R**2 * E^-`, `inverse_pft = E^+.T / (2*pi*R**2)`, checked by
  transforming an identity batch), then orthogonality, the non-unitarity figure, delta pair, generalized
  shift (a center bump shifted to a sample point), the four rules, starred-conjugate Parseval, rotation,
  linearity, zeroth harmonic, and the twisted conjugate symmetry `F_{-n} = (-1)^n conj(F_n)`.
- `08_visualization.ipynb` — only what no earlier notebook shows: the maze fixture, `plot_signal`'s
  color scales (gamma 0.3 `PowerNorm` vs. a linear scale, grayscale `SPATIAL_ANGULAR`, `twilight` phase
  pinned to `[-pi, pi]`), `forward_pft_traced`/`inverse_pft_traced` with `visualize_steps`/
  `visualize_pipeline`, the `PFTTrace` fields, `PFTTrace.save` to a temp dir, and `close()`.

- `09_experimental_data.ipynb` — measured radial k-space: a lines-to-spokes figure, `stored_arrays`/
  `load_stored_array` then `selectable_axes`/`select_radial_kspace` on the committed MD-CNN excerpt as a single sample and as a coil batch, a log-magnitude
  plot of the uniform polar k-space (the asymmetric echo's zeros visible), a `PolarGrid(128, 98, R=208)`
  checked against the data radius, the exact path (`resample_uniform_polar_frequency` -> `inverse_pft`, coil
  root-sum-of-squares) next to the labeled ring-consistent route (`inverse_pft_ring(..., radius=...)`, ~15 s),
  `radial_lines_to_polar` on synthetic interleaved/golden-angle lines, and `RadialKSpace.save` to a
  `tempfile.mkdtemp` directory plus a description of the terminal interface.
- `10_terminal_user_interface.ipynb` — `python -m pypft`/`pypft`, the screen flow, input formats, on-grid
  vs. uniform polar input, `output_filename`'s naming, then `PyPFTApp` driven headlessly with top-level
  `async with app.run_test()` on the maze (saving into a `tempfile.mkdtemp` `output_directory`), printing
  what the screens said only after the block (Textual captures `print` while the app runs), and showing
  one saved PNG.

Internal-plumbing work (the DHT's N-D generalization, the angular DFT subsystem) deliberately gets no
notebook of its own. Notebooks link to each other with plain Markdown links to the `.ipynb` file
(`[02_sampling_grids](02_sampling_grids.ipynb)`); `myst_nb` resolves them in the docs build, and a link to a
missing notebook fails `sphinx-build -W`. See "Notebook citations and `CITATION.cff`" above for the
citation format — `02_sampling_grids`/`05_pft_and_ipft` cite Yao & Baddour Part II; `09_experimental_data` cites the MD-CNN
dataset and its paper (both in `CITATION.cff`); `03_discrete_hankel_transform`
cites Baddour's DHT chapter; `07_pft_properties` cites the Part I paper; `00`/`01`/`04`/`06`/`08`/`10` cite
nothing.

**Notebook style lints** (`CONTRIBUTING.md`, "Notebook style"): `tests/test_notebook_style.py` (no
hard-wrapped Markdown — two consecutive prose lines, or a list item followed by a prose line, fail; no
backticked span containing `^` or `_{`; a figure-drawing notebook's first code cell starts, after comments,
with `%matplotlib inline`) and `tests/test_no_backend_mentions.py` (no `cv2`/`cv`/`cv_`/
`OpenCV`/`warpPolar`/`remap` in any cell; also scans `README.md`, `CONTRIBUTING.md`, `docs/*.rst`, but
**never `src/` docstrings** — the developer's call; the `opencv-python-headless` distribution name is
exempt) apply to every notebook under `notebooks/` (`NOTEBOOK_PATHS` in `tests/conftest.py`, a glob). Rewritten notebooks also comment every
code-cell step, pass arguments by keyword (`ax.imshow(X=...)`, `np.max(a=...)`), label both axes of every plot showing a quantity, and use
American spelling (`center`, `toward`). `08_visualization` saves its trace into a persistent
`output_directory` (`tempfile.mkdtemp`) the reader is told to change.

## Conventions (from `CONTRIBUTING.md`)

- **Constant extraction**: extract constants as much as possible — module-level constants go at the top of
  the module; anything shared more broadly goes at whatever level actually shares it.
- **Errors/warnings**: raise exceptions for invalid state — prefer purpose-built subclasses of builtin
  exceptions (e.g. `Val1ValueError(ValueError)`) over generic ones when callers need to distinguish cases;
  use the `warnings` module (not silent fallback) when degrading gracefully instead of raising.
- **Type annotations**: required on all function signatures, with input validation via the
  `*Validator` classes above.
- **Keyword arguments**: pass arguments by keyword at every call site where the callee's signature allows
  it — PyPFT's own API as well as third-party calls (`numpy`/`scipy`/`cv2`/etc.) — e.g.
  `PolarGrid(n_radial=383, n_angular=15, R=40.0)`, not `PolarGrid(383, 15, 40.0)`. Exempt only where the
  callee itself forces positional-only arguments (e.g. some NumPy/SciPy C-extension entry points).
- **Docstrings**: reStructuredText (PEP 287 / Sphinx `:param:`/`:type:`/`:raises:` style), not
  Google/NumPy style — see any method in `validators.py` for the pattern. Avoid Sphinx cross-reference
  roles (`:class:`/`:func:`/`:meth:`/`:data:`) — flake8's RST checker doesn't recognize them; use plain
  double-backtick literals (`` ``Name`` ``) instead, per the existing files. This also matters for
  `sphinx-build -W`: a module docstring written as plain indented prose rather than valid RST (inconsistent
  bullet-list indentation, missing blank lines around nested lists) raises docutils errors that `-W`
  escalates to build failures, even though `flake8-rst-docstrings` may not catch every case in isolation.
- **No historical narration**: docstrings, comments, and notebooks describe the current design and its
  rationale only — never "used to be," "no longer," "was removed/replaced," "an earlier revision," or
  similar before/after framing. `git log`/`git blame` is the record of what changed; source-level prose is
  not a changelog.
- **No explanations targeted at the Claude user, anywhere in the repo except this file**: source code, tests, and notebooks
  read as ordinary documentation for a human engineer — never as notes addressed to an AI coding assistant
  (bug-hunting narratives like "found by actually looking at a rendered image," process commentary like
  "written to fail without the fix," or any other meta-commentary about how or why a change was made). A
  general technical explanation with lasting value (a numerical-instability warning, a design-choice
  rationale) belongs in `DESIGN_NOTES.md` at the repo root, referenced by file and section from the code it
  explains — not inlined. A one-off changelog-style note needed only while actively working belongs in a
  scratch file under `.local_files/` (gitignored), never committed into tracked source.
- **Line length**: 88 chars everywhere — `pyproject.toml`'s `[tool.black]`/`[tool.flake8]`, `README.md`,
  and `.vscode/settings.json` (editor rulers, `rewrap.wrappingColumn`) all agree.
- **Code-sectioning comments**: `# ` + repeated character to fill the line width — `=` for a top-level
  section, `-` for a subsection, `*`/`.` for the same one indent level in. Snippets: `@ section`,
  `@ subsection`, `@ isection`, `@ isubsection` in `.vscode/helpers.code-snippets`.

## Benchmarking

`benchmarks/` (tracked) holds `bench_dht.py`/`bench_dft.py`/`bench_pft.py` (pytest-benchmark test functions
comparing the DHT/DFT/PFT-batching implementations, respectively) and `run_dht_benchmarks.py`/
`run_dft_benchmarks.py`/`run_pft_benchmarks.py` (run them and export a sorted Markdown report each); see the
DHT/DFT/PFT architecture sections above for how their results drove each subsystem's own
`DEFAULT_IMPLEMENTATION`/`DEFAULT_PFT_IMPLEMENTATION`.

**A benchmark result never justifies deleting an implementation.** `DHTImplementation.VECTORIZED`
(`src/pypft/dht/_vectorized.py`) has lost every DHT benchmark ever run against it, including the batched
`(radial, angular, batch)` regime — see its own `DEFAULT_IMPLEMENTATION` docstring for the numbers — and it
is still shipped. If a strategy measures slower than its siblings, say so in the relevant
`DEFAULT_*_IMPLEMENTATION`/benchmark docstring and *suggest* removing it in the phase/PR report; do not
delete the module, the enum member, or its `[project]` dependency (e.g. `numba`) unilaterally. Removing a
working implementation is the developer's call to make, not an automatic consequence of a slow number.

## Local, gitignored data

`.local_files/` (gitignored) holds local scratch/reference material for development, not part of the
package or test suite:

- `sources/` — reference papers on the DHT, the polar-coordinate DFT/PFT, and Bessel functions (Baddour
  2019's DHT book chapter and Mathematics Part I paper, Yao & Baddour's PeerJ CS Part II paper and its
  supplementary appendix, `bessel_properties.md`, a distillation of Bessel-function recurrence/derivative
  relations — retained for reference even though nothing in `src/` currently derives a kernel via Bessel
  recurrence (see `DESIGN_NOTES.md`, "DHT: the kernel's Bessel values must be computed directly, never via
  order recurrence") — and `pft_properties.md`, an
  equation-level distillation of both papers' operational rules, mirroring `bessel_properties.md`'s own
  style). These are the sources the notebooks' References cells and `CITATION.cff` cite.
- `benchmarks/results/` — timestamped Markdown reports generated by `benchmarks/run_dht_benchmarks.py`/
  `run_dft_benchmarks.py`/`run_pft_benchmarks.py` (gitignored since they're generated artifacts, not
  source, even though the scripts that produce them are tracked in the top-level `benchmarks/` directory).
