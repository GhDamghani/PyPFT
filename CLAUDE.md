# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

PyPFT is a Polar Fourier Transform toolkit for reconstructing polar-coordinate MR images: it chains an
angular FFT (`numpy.fft.fft`), a Hankel transform (not available elsewhere in the Python ecosystem, hence
this package), and an inverse angular FFT — see `README.md` for the math and the underlying paper.

The package was substantially rewritten from scratch in the `start over` commit (deccca8), which deleted a
full prior layout (`backends/`, `cli/`, `core/`, `dft/`, `dht/`, `fields/`, `grids/`, `idft/`, `io/`, docs,
benchmarks, notebooks, release scripts) and is being rebuilt incrementally, one reviewable unit at a time.
So far: `src/pypft/dht/` (the
discrete Hankel transform, three strategy implementations), `src/pypft/dft/` (the angular discrete Fourier
transform, two strategy implementations), `src/pypft/utils/validators.py` (shared input validation),
`src/pypft/axes.py` (the axis vocabulary and centered-angular convention), `src/pypft/geometry.py` (the
Cartesian↔polar image bridge), `src/pypft/grid.py` (`PolarGrid`, the transform's own order-dependent
sampling grid, plus the production `sample_cartesian` sampler and the `check_adequacy`/
`check_nyquist_adequacy` warnings), `src/pypft/references.py` (citation machinery), `src/pypft/transform.py`
(`forward_pft`/`inverse_pft`, the full PFT/IPFT pipeline, and the `scaled_hankel` step underlying both),
`src/pypft/domains.py` (`Domain` and the four `BaseSignal` subclasses, a typed shell over the PFT chain),
and a Sphinx docs skeleton (`docs/`) with five tutorial notebooks. There are still no batching,
visualization, or a CLI. Do not assume any prior architecture, module, or API still exists — check the
current file tree before referencing paths from git history.

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
- Benchmark the DHT/DFT implementations: `uv run python benchmarks/run_dht_benchmarks.py` /
  `uv run python benchmarks/run_dft_benchmarks.py` — neither is part of `uv run pytest` (they live outside
  `testpaths`), since they're dev tooling, not a correctness check. Both write timestamped Markdown reports
  to the gitignored `.local_files/benchmarks/results/`.
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
the DHT API, the domains API — `Domain`, `BaseSignal`, and its four subclasses — the geometry functions,
the grid API — `PolarGrid`, `LimitKind`, `sample_cartesian`, `check_adequacy`, `check_nyquist_adequacy` —
`Reference`/`cite`/`bibliography`, and `forward_pft`/`inverse_pft`), listed in `__all__` — this is what
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
  `_type_is_polar_grid`, likewise lives in `src/pypft/grid.py`). `NumpyValidator` also has
  `value_is_2d`/`value_is_finite` (added for `pypft.grid.sample_cartesian`'s image argument), alongside the
  pre-existing `value_is_1d`/`value_is_at_least_1d`, and `value1_shape_matches_value2` (added for
  `pypft.transform.forward_pft`/`inverse_pft`'s whole-array shape check against a `PolarGrid`), alongside
  the pre-existing single-axis `value1_axis_length_matches_value2`.
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

- **`forward_pft`/`inverse_pft` follow PyPFT's own `(radial, angular)` axis layout** (`pypft.axes.Axis`),
  matching `pypft.geometry.cartesian_to_polar`'s convention — **not** `PolarGrid.r`'s/`sample_cartesian`'s
  own `(angular, radial)` layout (each row of `PolarGrid.r` is one harmonic's own radial samples, the
  natural shape for *building* the grid, not for storing a transformed image). A `sample_cartesian` result
  must be transposed before it is passed to `forward_pft`.
- **`scaled_hankel(values, grid, *, direction, axis)`** is the per-harmonic step underlying both directions:
  it loops over `grid.harmonics` (one `hankel_transform`/`inverse_hankel_transform` call per harmonic, since
  each order needs its own kernel) along the axis complementary to `axis`, and is the single place in
  `src/` that names the radial axis — `forward_pft`/`inverse_pft` always pass `Axis.RADIAL` explicitly
  rather than relying on `hankel_transform`'s own unrelated `axis=-1` default.
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

A typed, optional shell over the already-verified `forward_pft`/`inverse_pft` chain — it introduces no
new numerics, only a way to name where a polar array sits along that chain and to walk between those
points one verified step at a time. Array-in/array-out (`pypft.transform`) stays the primitive; nothing
in `pypft.transform`/`pypft.grid` requires wrapping a signal in one of these classes.

- **`Domain(Enum)`** has exactly four members on one ordered path, `_CHAIN`: `SPACE_POLAR ->
  SPACE_HARMONIC -> FREQUENCY_HARMONIC -> FREQUENCY_POLAR`. Word 1 of a member's name (`SPACE`/
  `FREQUENCY`) is the radial coordinate, changed only by the DHT (edge 1 of `_CHAIN`); word 2 (`POLAR`/
  `HARMONIC`) is the angular coordinate, changed only by the angular DFT/IDFT (edges 0 and 2). Because
  this is a path graph with no branches, a transition is legal exactly when it moves one step along
  `_CHAIN` — there is deliberately **no separate `_LEGAL_MOVES` table**, which would just encode that
  same adjacency a second time.
- **`BaseSignal`** is a frozen dataclass carrying `values`/`grid` plus a `domain: ClassVar[Domain]` fixed
  per subclass; `__post_init__` validates `values`'s type and its shape against `grid`, reusing the same
  pattern as `pypft.transform`'s own `_validate_pft_input`.
- **Four subclasses, one per `Domain` member** (`SpacePolarSignal`, `SpaceHarmonicSignal`,
  `FrequencyHarmonicSignal`, `FrequencyPolarSignal`), each defining only the step methods to its own
  neighbours in `_CHAIN`: `to_harmonics`/`to_angles` for the angular DFT/IDFT edges, `to_frequency`/
  `to_space` for the DHT edge. A step method's own implementation is a thin wrapper around
  `pypft.dft.angular_dft`/`inverse_angular_dft` or `pypft.transform.scaled_hankel` — exactly the calls
  `forward_pft`/`inverse_pft` themselves make, so a hand-written chain of step calls matches those
  functions bit-for-bit. Calling a step method a subclass does not define (e.g.
  `SpaceHarmonicSignal.to_harmonics`, which only `SpacePolarSignal` has) is therefore a `pyright` error on
  a hand-written chain, not just a runtime `AttributeError`.
- **`BaseSignal.to(domain)`** is the dynamic counterpart: a short loop indexing into `_CHAIN` plus the
  parallel `_STEP_TOWARD`/`_STEP_BACKWARD` tuples (which method advances/retreats across each of the
  chain's three edges), calling the matching named step method via `getattr` until `domain` is reached.
  It validates its own argument is a real `Domain` member (`EnumValidator`), which is the only way a call
  to `to` can fail — every pair of `Domain` members is reachable along the path graph, so there is no
  notion of an "illegal" domain pair once the argument itself is valid.

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
  ~11.7us vs. ~12.1us). A third `SCIPY_WORKERS` (`workers=-1`) implementation was considered for `SCIPY`'s
  ~33% win on batched, non-trailing-axis input, and rejected: explicit worker parallelism measured only
  ~2% faster than `SCIPY`'s own default in that regime, so the win is the algorithm, not parallelism — not
  worth a third strategy. See the constant's own docstring and the benchmark report for the full numbers.
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
is non-uniform, the central gap that never fully closes, the angular-vs-radial resolution trade-off via
`check_adequacy`, and the Nyquist condition via `check_nyquist_adequacy`), `03_pft_and_ipft.ipynb` (the
full `forward_pft`/`inverse_pft` chain against the Gaussian oracle, ending with the dB-error map reproducing
Yao & Baddour Part II's own published figure), and `05_domains.ipynb` (`Domain`/`BaseSignal`: walking the
PFT's chain by hand one verified step at a time vs. dynamically via `to`, and the hand-written illegal-step
`pyright`/`AttributeError` pair) exist so far. Internal-plumbing work (the DHT's N-D generalization, the
angular DFT subsystem) deliberately gets no notebook of its own — their gate is that every *existing*
notebook still executes, since a notebook per internal subsystem would duplicate the API reference without
teaching a workflow. See "Architecture: citations" above for the citation discipline notebooks must follow
when they state a mathematical result — `02_sampling_grids` is the first notebook to actually cite anything
(`YaoBaddour2020`); `05_domains` introduces no new numerics, so it cites nothing.

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

`benchmarks/` (tracked) holds `bench_dht.py`/`bench_dft.py` (pytest-benchmark test functions comparing the
DHT/DFT implementations, respectively) and `run_dht_benchmarks.py`/`run_dft_benchmarks.py` (run them and
export a sorted Markdown report each); see the DHT/DFT architecture sections above for how their results
drove each subsystem's own `DEFAULT_IMPLEMENTATION`.

## Local, gitignored data

`.local_files/` (gitignored) holds local scratch/reference material for development, not part of the
package or test suite:

- `sources/` — reference papers on the DHT, the polar-coordinate DFT/PFT, and Bessel functions (Baddour
  2019's DHT book chapter and Mathematics Part I paper, Yao & Baddour's PeerJ CS Part II paper and its
  supplementary appendix, and `bessel_properties.md`, a distillation of Bessel-function recurrence/
  derivative relations — retained for reference even though the `_recurrence.py` implementation that used
  it was removed for numerical instability; see the DHT architecture section above). These are the sources
  `src/pypft/references.py`'s `Reference` members cite.
- `benchmarks/results/` — timestamped Markdown reports generated by `benchmarks/run_dht_benchmarks.py`/
  `run_dft_benchmarks.py` (gitignored since they're generated artifacts, not source, even though the
  scripts that produce them are tracked in the top-level `benchmarks/` directory).
