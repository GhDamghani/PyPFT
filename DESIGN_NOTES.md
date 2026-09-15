# Design Notes

Technical rationale and pitfall warnings for PyPFT that are too detailed for an inline comment but have
lasting value for anyone changing the referenced code. Source code, `CLAUDE.md`, and the test suite point
here by section heading rather than repeating this content inline.

## Packaging: `dependencies` must precede `[project.urls]`

`pyproject.toml`'s `[project]` table must keep `dependencies = [...]` before `[project.urls]`. TOML
attaches a bare `dependencies` key to whichever table header precedes it, so if `[project.urls]` comes
first, `dependencies` silently attaches to `[project.urls]` instead of `[project]`. `numba`/`scipy` then
vanish from the resolved lock, and `uv sync` fails with a setuptools `project.urls.dependencies`
validation error — a confusing symptom for what is really a table-ordering mistake.

## DHT: the kernel's Bessel values must be computed directly, never via order recurrence

`src/pypft/dht/`'s kernel construction (`BaseDHT._bessel_kernel`) always calls `scipy.special.jv`
directly, once per order — never via the three-term Bessel order recurrence. That recurrence is
exponentially unstable once the order exceeds the argument, which the kernel evaluates by construction: a
recurrence-built kernel measures `max|Y^{nN} @ Y^{nN} - I|` at `2.1e+16` by order 47 (size 64), against
`~7e-6` for the direct/cached kernel at the same order. `tests/dht/conftest.py`'s `DHT_ORDERS` includes
high orders (16, 32, 64), not just low ones, specifically to keep this class of numerical instability
visible; `tests/dht/tolerance.py`'s `dht_tolerance(order, size)` model (rather than a flat tolerance)
exists for the same reason — a flat bound is either too loose to catch this failure mode or too tight to
accept the correct kernel above ~order 24.

## DHT: kernel formulation is `Y^{nN}`, not the symmetric `T^{nN}`

The DHT uses Baddour's `Y^{nN}` formulation (paper's Eq. 39), not the alternative symmetric `T^{nN}`
formulation (Eq. 44). Both are self-inverse (`M @ M = I`), but `T^{nN}` only preserves Parseval's theorem
on Sec. 7's "scaled" vectors, not on raw `f`/`F` values directly, and produces wrong (sign-oscillating)
results when checked against a known continuous Hankel-transform pair (a self-reciprocal Gaussian,
`tests/dht/test_gaussian.py`). `Y^{nN}` is used so the public API works correctly in terms of raw signal
values — see `src/pypft/dht/_base.py`'s module docstring for the full rationale.

## DFT: `SCIPY_WORKERS` is not offered as a separate strategy

`DFTImplementation.SCIPY`'s ~33% win over `NUMPY` on batched, non-trailing-axis input is attributable to
`scipy.fft`'s algorithm, not parallelism: explicit worker parallelism (`workers=-1`) measures only ~2%
faster than `SCIPY`'s own default in that regime. A dedicated `SCIPY_WORKERS` strategy would add a third
implementation for a ~2% gain, which isn't worthwhile.

## PFT: `STACKED_KERNEL` has no separate `PARALLEL` strategy

`PFTImplementation.STACKED_KERNEL`'s performance win comes from a single BLAS call (`numpy.matmul` across
every harmonic and batch element at once), not from added parallelism — it is BLAS-bound, not
Python-overhead-bound, so a `numba`-parallelized `PARALLEL` strategy would have nothing to win against.

## PFT: `STACKED_KERNEL`'s kernel-stack cache is required for its own performance win

`scaled_hankel`'s `STACKED_KERNEL` strategy caches its per-harmonic kernel stack (an `lru_cache` keyed on
the hashable `(grid, direction)`, bounded by `STACKED_KERNEL_CACHE_MAXSIZE`). Rebuilding and copying the
whole stack on every call makes `STACKED_KERNEL` measure *slower* than `HARMONIC_LOOP` even on the batched
`(radial, angular, batch)` workload it is otherwise chosen for — the cache is load-bearing for
`STACKED_KERNEL` being the default at all.

## Visualization: phase color range is pinned to `[-pi, pi]`

Every phase `imshow` call in `src/pypft/viz.py` pins `vmin`/`vmax` to `[-pi, pi]` explicitly, rather than
leaving the range to `matplotlib`'s own auto-scaling. Without an explicit `vmin`/`vmax`, `matplotlib`
auto-scales a phase plot's color range from the actual data — fine for a signal whose phase genuinely
varies, but a real-valued signal's phase is *exactly* `0.0` at every pixel, which drives
`matplotlib.colors.Normalize` into its degenerate `vmin == vmax` case. That case maps every value to
normalized `0.0`, **not** the midpoint `0.5` a naive reading would expect — so a perfectly flat phase
array renders as `twilight`'s pale wrap-around endpoint (`cmap(0.0)`, RGB `(226, 217, 226)`) instead of
the true, correctly-scaled color for that phase value (`cmap(0.5)`, RGB `(47, 20, 54)` at phase 0). The
same degenerate path is hit by *any* exactly-constant phase, not just zero, silently discarding whatever
the constant value actually was.

## Visualization: `render_cartesian`'s `grid.theta` must be negated

`render_cartesian` computes Cartesian coordinates as `y = -grid.r * np.sin(grid.theta)` — the negation is
required, not optional. `grid.theta` follows `pypft.geometry`'s image-coordinate convention (increasing
angle rotates toward increasing row, i.e. downward on screen), but `imshow`'s own `origin="lower"` expects
`y` to increase upward. A bare `np.sin(grid.theta)` without the negation renders every image upside down.
