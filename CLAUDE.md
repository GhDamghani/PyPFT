# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying paper.

PyPFT is built incrementally, one reviewable unit at a time. So far: `src/pypft/dht/` (the
discrete Hankel transform, three strategy implementations), `src/pypft/dft/` (the angular discrete Fourier
transform, two strategy implementations), `src/pypft/utils/validators.py` (shared input validation),
`src/pypft/axes.py` (the axis vocabulary and centered-angular convention), `src/pypft/geometry.py` (the
Cartesian↔polar image bridge), `src/pypft/grid.py` (`PolarGrid`, the transform's own order-dependent
sampling grid, plus the production `sample_cartesian` sampler and the `check_adequacy`/
`check_nyquist_adequacy` warnings), `src/pypft/references.py` (citation machinery), `src/pypft/transform.py`
(`forward_pft`/`inverse_pft`, the full PFT/IPFT pipeline, the `scaled_hankel` step underlying both -- two
strategy implementations, plus 3-D `(radial, angular, batch)` support on top of the plain 2-D case),
`src/pypft/domains.py` (`Domain`/`BaseSignal`, the typed legal-move shell over `transform.py`'s numerics),
`src/pypft/_kernel.py` (a private, from-scratch `O(N**4)` oracle reproducing `forward_pft`/`inverse_pft`,
used only by `tests/test_kernel.py`), `src/pypft/viz.py` (`plot_signal`/`BaseSignal.plot`,
`render_cartesian`, and `forward_pft_traced`/`inverse_pft_traced` -- `Axes`/`Figure`-based visualization,
never writing to disk), analytical-property test suites for both transforms
(`tests/dht/test_kernel_properties.py`, `tests/test_transform_properties.py`), and a Sphinx docs skeleton
(`docs/`) with seven tutorial notebooks. There is still no CLI. Do not
assume any prior architecture, module, or API still exists — check the current file tree before referencing
paths from git history; module names like `backends/`, `cli/`, `core/`, `fields/`, `grids/`, `idft/`, `io/`
may still appear in older commits but do not correspond to anything in the current tree. `CHANGELOG.md`
does not exist and must not be recreated without checking with the developer first, even though some commit
messages still follow the Keep-a-Changelog convention. `DESIGN_NOTES.md` (repo root) holds general
technical rationale and pitfall warnings referenced by file/section from source and from this file's own
Architecture sections below — see the Conventions section for what belongs there.

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
- Format/lint/type-check: `uv run black src tests benchmarks scripts`,
  `uv run isort src tests benchmarks scripts`, `uv run flake8 src scripts` (only `src` and `scripts` are
  linted — `--extend-select=D1` in `pyproject.toml` enforces missing-docstring checks via
  `flake8-docstrings`), `uv run pyright`, `uv run vulture src scripts`.
- Run the whole quality gate at once (what CI runs): `./scripts/Invoke-QualityGate.ps1` — pytest, the
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
- Regenerate a polar-sampled test fixture: `uv run python scripts/make_test_image.py --input <path>
  --output <path> [--size 256] [--n-radial 1024] [--n-angular 127] [--r-fraction 0.95]` — rasterizes a
  source image (vector `.eps`/`.ps` via Pillow's Ghostscript-backed `EpsImagePlugin`, or an ordinary raster
  format directly), resizes it to a square, and samples that square via `pypft.sample_cartesian` onto a
  `pypft.PolarGrid(n_radial, n_angular, R)`, writing the result as an uncompressed 8-bit grayscale TIFF.
  `tests/samples/hedge_maze.tif` (the `Domain.SPACE_POLAR` sample signal in
  `notebooks/07_visualization.ipynb`) was produced this way; see `THIRD-PARTY-NOTICES.md` for that file's
  own third-party attribution, separate from this project's own BSD-3-Clause `LICENSE`.
- `pytest.ini_options` sets `filterwarnings = ["error"]` — any `warnings.warn` in `src/` needs a matching
  `pytest.warns` test, or the suite fails.

`pyproject.toml`'s `[project]` table must keep `dependencies = [...]` before `[project.urls]` — see
`DESIGN_NOTES.md`, "Packaging: `dependencies` must precede `[project.urls]`" for why.

## CI

`.github/workflows/ci.yml` runs on every pull request and on push to `main`, across a
`windows-latest`/`ubuntu-latest`/`macos-latest` matrix (`shell: pwsh` throughout): `astral-sh/setup-uv`,
`uv sync`, then `scripts/Invoke-QualityGate.ps1` alone — it already runs `scripts/Test-Notebooks.ps1` as
one of its own steps, so CI does not call it a second time. No inline shell logic lives in the YAML — both
scripts are meant to be run locally too, so a red CI leg is always reproducible with one local command.

## Architecture: the package facade

`src/pypft/__init__.py` re-exports the public surface from each submodule (`Axis`, `DEFAULT_BATCH_AXIS`,
the DHT API, the domains API — `Domain`, `BaseSignal`, and its four subclasses — the geometry functions,
the grid API — `PolarGrid`, `LimitKind`, `sample_cartesian`, `check_adequacy`, `check_nyquist_adequacy` —
`Reference`/`cite`/`bibliography`, `forward_pft`/`inverse_pft`, and the visualization API —
`PFTTrace`, `forward_pft_traced`, `inverse_pft_traced`, `plot_signal`, `render_cartesian` — listed
in `__all__` — this is what
keeps `flake8`'s unused-import check (`F401`) satisfied for a pure re-export module. `pypft.dft` is deliberately **not** re-exported here: it is internal plumbing
between the geometry/axes layer and the DHT (no notebook of its own — see the Notebooks section), reachable
as `pypft.dft.angular_dft` and documented via `docs/api.rst`, the same way `pypft.utils.validators` is
public but un-re-exported. `pypft.grid`'s two warning classes (`AdequacyWarning`, `NyquistWarning`) and
`pypft.transform`'s `Direction`/`scaled_hankel` are likewise reachable via `pypft.grid.*`/`pypft.transform.*`
but not re-exported at the top level: the warnings because filtering on them is an opt-in developer action
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
  against a `PolarGrid`), and `value_has_ndim_in(value, ndims)` (general-purpose rank check) with
  `value_is_2d_or_3d` (built on it) for the polar layer's optional trailing batch axis —
  `pypft.transform.scaled_hankel`/`forward_pft`/`inverse_pft` and `pypft.domains.BaseSignal` all accept
  2-D or 3-D `values`, batch axis last. `MatplotlibValidator` (`type_is_axes`, `type_is_figure`) covers
  `pypft.viz`'s `Axes`/`Figure` arguments and its `PFTTrace.figures` field, ordered before `NumpyValidator`
  (third-party classes are alphabetical: `matplotlib` before `numpy`).
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
to the batch axis" coincide exactly where that's unambiguous. Low-level generic transforms (`pypft.dht`,
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

`cartesian_to_polar`/`polar_to_cartesian` wrap `cv2.warpPolar` (forward and its inverse-map mode) to
resample an ordinary image onto, and back off of, a **uniform** polar grid. This is explicitly *not* the
discrete Hankel transform's own sampling grid (which is order-dependent and non-uniform, per Baddour's
`r_nk`) — these two functions exist because `warpPolar` is the natural first illustration of what "polar"
means for an image, not because their output feeds the transform. The real, order-dependent sampler is
`pypft.grid.sample_cartesian` (see the sampling-grid section below).

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
- **`LimitKind.SPACE_LIMITED` vs. `BAND_LIMITED`** (PeerJ CS Part II, Eqs. 14-17) are the same
  Bessel-zero ratios with `r`/`rho` swapped and `R` reinterpreted as the band limit `Wr` — implemented as
  exactly that: `PolarGrid.r`/`.rho` pick one of `sample_points`'s two outputs for `SPACE_LIMITED` and the
  other for `BAND_LIMITED`, rather than a second formula.
- **`check_adequacy(grid)`** warns (`AdequacyWarning`, never raises) when `n_radial` is too small for
  `n_angular`, using a log-log least-squares fit of this package's own measured forward-Gaussian-oracle
  error to `(n_angular, n_radial)` (nine points, residual under 0.6 dB) — *not* a formula from either
  paper. The warning threshold (`-60` dB predicted average error) is chosen to match the eventual PFT
  pipeline's own forward-accuracy acceptance gate, so the two stay consistent.
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
  (`pypft.axes.Axis`), matching `pypft.geometry.cartesian_to_polar`'s convention — **not**
  `PolarGrid.r`'s/`sample_cartesian`'s own `(angular, radial)` layout (each row of `PolarGrid.r` is one
  harmonic's own radial samples, the natural shape for *building* the grid, not for storing a transformed
  image). A `sample_cartesian` result must be transposed before it is passed to `forward_pft`. Both accept
  an optional trailing batch axis (a 3-D array) on top of the plain 2-D case — costs no new numerical
  kernel, since the angular DFT/IDFT and the DHT already operate along one named axis of an otherwise
  arbitrary-rank array. `batch_axis: int = DEFAULT_BATCH_AXIS` is keyword-only on both (`pypft.axes`'s
  axis-default tiering: only the batch axis is ever defaulted); passing it on a 2-D array, or naming
  anything other than a 3-D array's last axis, raises `ValueError` — PyPFT's own layout always places the
  batch axis last, there is no support for putting it anywhere else.
- **`scaled_hankel(values, grid, *, direction, axis, angular_axis, implementation)`** is the per-harmonic
  step underlying both directions: it loops over `grid.harmonics` (one Hankel transform per harmonic, since
  each order needs its own kernel), and is the single place in `src/` that names the radial axis —
  `forward_pft`/`inverse_pft` always pass `Axis.RADIAL`/`Axis.ANGULAR` explicitly rather than relying on
  `hankel_transform`'s own unrelated `axis=-1` default. `angular_axis` is a
  required argument (not derived as "the other axis") since a 3-D input has more than two axes to choose
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

## Architecture: typed domain objects (`src/pypft/domains.py`)

`Domain(Enum)` names the four points a polar array occupies across the PFT/IPFT chain
(`SPACE_POLAR`/`SPACE_HARMONIC`/`FREQUENCY_HARMONIC`/`FREQUENCY_POLAR`); word 1 of each member is the
radial coordinate (changed only by the DHT), word 2 is the angular coordinate (changed only by the angular
DFT/IDFT). `BaseSignal` (frozen dataclass: `values`, `grid`, `domain: ClassVar[Domain]`, `batch_axis`) and
its four subclasses (`SpacePolarSignal`, `SpaceHarmonicSignal`, `FrequencyHarmonicSignal`,
`FrequencyPolarSignal`) are a thin, optional wrapper over `pypft.transform`'s already-verified numerics —
`values`-in/`values`-out through `forward_pft`/`inverse_pft`/`scaled_hankel`/`angular_dft` remains the
primitive the numeric path never requires this module for. Key points:

- **One ordered `_CHAIN` tuple, no `_LEGAL_MOVES` table.** The chain is a path graph (no branches, no
  cycles), so a transition is legal exactly when it moves one step along `_CHAIN` — encoding legality
  separately would just duplicate it. Each subclass defines only the step methods for its own neighbours
  (destination-named: `to_harmonics`/`to_angles`/`to_frequency`/`to_space`), so calling a step method that
  does not exist on a given subclass is a `pyright` error on a hand-written chain, not just a runtime
  `AttributeError`. `to(domain)` is the dynamic counterpart: a 5-line walk along `_CHAIN`, never a general
  graph search, since the only decision at each step is which direction and which of `_STEP_TOWARD`/
  `_STEP_BACKWARD` advances one edge that way.
- **`batch_axis: int = DEFAULT_BATCH_AXIS`** mirrors `forward_pft`/`inverse_pft`'s own parameter exactly —
  `BaseSignal.__post_init__` reuses `pypft.transform._validate_pft_input` directly (the same 2-D-or-3-D,
  batch-axis-must-be-last validation) rather than duplicating it, and every step method threads
  `self.batch_axis` through to the signal it returns, since batching never changes along the chain.

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
  meaningful). Every `Domain` member is treated as complex-valued: `SPACE_HARMONIC` is an angular
  DFT's own coefficients, generically complex even when the space-domain signal being transformed is real
  (a DFT of real input is only symmetric, not real, in general); `SPACE_POLAR` can likewise carry a
  non-trivial phase after a full forward-then-inverse round trip. Since
  every domain takes the same path, `plot_signal` always returns `tuple[Axes, Axes]`, and there is no
  `_plot_complex_signal`/`_plot_magnitude_signal` helper split, since there is no dispatch for such a
  helper to support.
  `plot_signal` still type-validates a given `ax` as a 2-tuple *before* unpacking it into
  `(magnitude_ax, phase_ax)` — an unguarded unpack would raise `ValueError` instead of `TypeError` on a
  wrong-length `ax`, breaking `plot_signal`'s own documented type-validation contract. `BaseSignal.plot()`
  is a thin delegate to this function, using a **function-local** import of `pypft.viz` (`domains.py`
  cannot import it at module level: `pypft.viz` itself imports `Domain`/`BaseSignal` from `pypft.domains`,
  so a top-level import would be circular).
- **`_magnitude_cmap(domain)` grayscales `SPACE_POLAR`'s own magnitude specifically**, everywhere a
  magnitude is drawn (`plot_signal`, `render_cartesian`) — a
  `SPACE_POLAR` magnitude is literally a photographic image (`pypft.grid.sample_cartesian`'s own image
  argument), unlike every other domain's more abstract Fourier/harmonic-coefficient magnitude, which stays
  at `matplotlib`'s own default colormap (`cmap=None`, resolved by each `imshow` call). Phase panels are
  never affected by domain — `cmap="twilight"` (a cyclic colormap, correct for any wrapped `[-pi, pi]`
  quantity regardless of domain) is unconditional, and so is its color *range*: `_PHASE_VMIN`/`_PHASE_VMAX`
  (`-np.pi`/`np.pi`) are always passed explicitly to every phase `imshow` call — see below for why.
- **Every phase `imshow` call pins `vmin`/`vmax` to `[-pi, pi]` explicitly**, rather than leaving the range
  to `matplotlib`'s own auto-scaling — see `DESIGN_NOTES.md`, "Visualization: phase color range is pinned
  to `[-pi, pi]`," for why (a degenerate-`Normalize` failure mode for any exactly-constant phase, e.g. a
  real-valued `SPACE_POLAR` signal). `tests/test_viz.py::test_plot_signal_phase_color_range_is_fixed_for_constant_phase`
  pins this down: `phase_ax.images[0].get_clim() == (-np.pi, np.pi)` must hold for a real-valued signal.
- **`render_cartesian(signal, *, height, width, ax=None)`** is the display-only counterpart for the two
  `POLAR` domains specifically (`SPACE_POLAR`/`FREQUENCY_POLAR` — the two whose angular axis is a physical
  angle, not a harmonic order): it interpolates `PolarGrid.r`/`.theta`'s own non-uniform sample points onto
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
  tuple of names (`"step_<domain>"`, `"pipeline"`) `PFTTrace.__post_init__` validates against `figures`' own
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
  Jupyter/`ipykernel` kernel, which already defaults to the inline backend, so no notebook needs this itself
  (see `notebooks/07_visualization.ipynb`).

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
executed and checked by `nbmake` in CI, not re-executed by the docs build.

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
tutorial sequence where each notebook assumes only its predecessors: `00_installation_and_quickstart.ipynb`,
`01_polar_and_cartesian_images.ipynb`, `02_sampling_grids.ipynb` (the `PolarGrid` sampling grid: why it
is non-uniform, the central gap that never fully closes, the angular axis's own harmonics and its
unpaired Nyquist bin for even `n_angular` via `grid.parity`, the angular-vs-radial resolution trade-off via
`check_adequacy`, and the Nyquist condition via `check_nyquist_adequacy`), `03_pft_and_ipft.ipynb` (the
full `forward_pft`/`inverse_pft` chain against the Gaussian oracle, ending with the dB-error map reproducing
Yao & Baddour Part II's own published figure), `04_transform_properties.ipynb` (a tour of both the DHT's
and the PFT's own analytical properties — self-inverse, the Kronecker-delta pair, the negative-order sign
relation, the generalized shift and its derived rules, kernel orthogonality, rotation equivariance,
linearity, and the DC term), `05_domains.ipynb` (typed domains and legal moves), `06_batches.ipynb`
(3-D `(radial, angular, batch)` support: exactness vs. looping the 2-D case, `BaseSignal`'s `batch_axis`,
the batch-axis-must-be-last validation, and a timing comparison against a Python loop), and
`07_visualization.ipynb` (`plot_signal`/`BaseSignal.plot`, `render_cartesian`, a full
forward-then-inverse round trip via `forward_pft_traced`/`inverse_pft_traced` with both
visualization keywords, and `PFTTrace.save`) exist so far.
Internal-plumbing work (the DHT's N-D generalization, the angular
DFT subsystem) deliberately gets no notebook of its own — their gate is that every *existing* notebook still
executes, since a notebook per internal subsystem would duplicate the API reference without teaching a
workflow. See "Architecture: citations" above for the citation discipline notebooks must follow when they
state a mathematical result — `02_sampling_grids` is the first notebook to actually cite anything
(`YaoBaddour2020`); `04_transform_properties` is the first to cite `Baddour2019a`/`Baddour2019b`.
`06_batches`/`07_visualization` state no new mathematical result (batching and visualization are engineering
properties, not new equations), so neither cites anything or needs a `bibliography(...)` cell.

## Conventions (from `README.md`)

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
- **No explanations targeted at the Claude user, anywhere in the repo**: source code, tests, and notebooks
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
  style). These are the sources `src/pypft/references.py`'s `Reference` members cite.
- `benchmarks/results/` — timestamped Markdown reports generated by `benchmarks/run_dht_benchmarks.py`/
  `run_dft_benchmarks.py`/`run_pft_benchmarks.py` (gitignored since they're generated artifacts, not
  source, even though the scripts that produce them are tracked in the top-level `benchmarks/` directory).
