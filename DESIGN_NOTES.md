# Design Notes

Technical rationale and pitfall warnings for PyPFT that are too detailed for an inline comment but have
lasting value for anyone changing the referenced code. Source code and the test suite point
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

## Grid: the spatial row index is a spoke, and the transform identifies it with a harmonic

`pypft.grid.PolarGrid` is the space-limited grid of Yao & Baddour, Part II (PeerJ CS 6:e257), Eqs.
14-15: $\theta_p = 2\pi p / N_2$, $r_{pk} = j_{|p|k} R / j_{|p|N_1}$, $\rho_{qm} = j_{|q|m} / R$. This
section records how it relates to an ordinary, uniform polar grid, why a uniform polar array must be
interpolated along each spoke before it is transformed, and why the average dB error alone cannot
certify accuracy. `tests/test_grid_relationship.py` pins each fact on small grids.

**Same spokes, different radii.** For equal `n_angular`, a uniform polar grid and `PolarGrid` have the
same uniform, centered spokes. They differ only along each spoke: a uniform grid uses the same radii on
every spoke; `PolarGrid` uses $r_{pk}$, which depends on the spoke. Away from the center the spacing
settles to about $\pi R / j_{|p|N_1}$, close to uniform; near the center it is not, there is no sample
at $r = 0$, and the innermost radius $j_{|p|1} R / j_{|p|N_1}$ grows with $|p|$. At `n_radial=382,
n_angular=15, R=40`, radial index 10 sits at $r = 1.124$ on spoke $p = 0$ and at $r = 1.458$ on spokes
$p = \pm 7$. A column of constant radial index is therefore not a ring; Part II itself notes that these
grids "are not true polar grids in the sense of equispaced sampling".

**The spatial row index is a spoke, not a harmonic.** Row `i` of `PolarGrid.r` holds the radii along
spoke $\theta_i$, sampled with Bessel order $|p_i|$ of the spoke's own angular sample index $p_i$.
Harmonics only exist after the angular DFT. The transform identifies the two indices: Part I
(Mathematics 7(8):698), Eq. 18, computes harmonic $n$ at $r = j_{nk} R / j_{nN_1}$ from samples taken at
$r = j_{pk} R / j_{pN_1}$, with $p$ the summation index, and calls this "a key assumption" of the
development, the one that permits the invertibility of the discrete transforms. Appendix A.6 compares it
with the conventional definition (Eq. A17), which samples harmonic $n$ at its own radii on every spoke
and does not yield an invertible square transform. The output side mirrors this: frequency-domain spoke
$q$, filed at $\rho_{qm}$, is assembled from harmonics $n$ evaluated at their own $\rho_{nm}$.
`PolarGrid.harmonics` is named after this identification; in the space domain it indexes spokes.

**Where the error comes from.** For the centered Gaussian $f = e^{-r^2}$ on the `n_radial=382,
n_angular=15, R=40` grid, the angular DFT of the on-grid samples has a spurious harmonic $\pm 1$ of
magnitude 0.060, and its harmonic 0 peaks at 0.939 instead of 1, because the samples of one radial index
sit at different radii. A circularly symmetric function has no harmonic 1 at all. The discrete Hankel
transform is not the problem: `hankel_transform` of the exact harmonic-0 samples $e^{-r_{0k}^2}$, scaled
by $2\pi$, matches $\pi e^{-\rho^2/4}$ to $1.8 \times 10^{-16}$ of its peak.

**Measurements.** A Gaussian centered at $\mathbf{x}_0$, $f = e^{-|\mathbf{x} - \mathbf{x}_0|^2}$, with
exact transform $\pi e^{-\rho^2/4} e^{-i \boldsymbol{\rho} \cdot \mathbf{x}_0}$ on the grid's own
$(\rho_{qm}, \psi_q)$. "On grid" samples $f$ at $(r_{pk}, \theta_p)$. "Uniform as-is" samples $f$ at
`n_radial` uniform radii strictly inside $(0, R)$, the same on every spoke, and feeds them to
`forward_pft` unchanged. "Uniform, spline" interpolates `n_radial + 1` uniform radii on $[0, R]$ along
each spoke onto `PolarGrid.r` with a cubic spline (`scipy.interpolate.CubicSpline`). "Ring route" is the
harmonic-consistent route described below, evaluated on seven spokes. `E_avg` is the mean of
$20 \log_{10}(|F - F_\text{exact}| / \max|F|)$ over all samples; "rel. L2" is
$\lVert F - F_\text{exact} \rVert / \lVert F_\text{exact} \rVert$.

| `n_radial` | `n_angular` | `R` | $\mathbf{x}_0$ | on grid `E_avg` | on grid rel. L2 | uniform as-is rel. L2 | uniform, spline rel. L2 | ring route rel. L2 |
| ---------- | ----------- | --- | -------------- | --------------- | --------------- | --------------------- | ----------------------- | ------------------ |
| 382        | 15          | 40  | $(0, 0)$       | −63.80 dB       | 0.243           | 0.071                 | 0.243                   | 4.3e-16            |
| 382        | 15          | 40  | $(2, -1)$      | −86.62 dB       | 0.244           | 0.242                 | 0.244                   | 2.0e-2             |
| 764        | 15          | 40  | $(2, -1)$      | −106.43 dB      | 0.239           | 0.237                 | 0.239                   | 2.0e-2             |
| 382        | 15          | 80  | $(2, -1)$      | −64.31 dB       | 0.172           | 0.175                 | 0.172                   | 1.9e-2             |
| 764        | 15          | 80  | $(2, -1)$      | −86.49 dB       | 0.140           | 0.138                 | 0.140                   | 1.9e-2             |
| 200        | 63          | 10  | $(0, 0)$       | −68.23 dB       | 1.07            | 1.73                  | 1.07                    | 1.0e-15            |
| 200        | 63          | 10  | $(2, -1)$      | −98.20 dB       | 2.16            | 2.19                  | 2.16                    | 4.6e-13            |
| 400        | 63          | 20  | $(2, -1)$      | −96.26 dB       | 1.43            | 1.45                  | 1.43                    | 3.4e-13            |
| 800        | 63          | 40  | $(2, -1)$      | −95.33 dB       | 0.928           | 0.928                 | 0.928                   | 2.7e-13            |

What the table shows:

- **On-grid samples carry an error of 0.14 to 2.2 in relative L2**, against `E_avg` values between −64
  and −106 dB. Doubling `n_radial` at fixed `R` improves `E_avg` by about 20 dB and leaves the relative
  L2 error essentially unchanged (0.244 to 0.239): more radial samples cannot remove it.
- **The error falls as `R` grows relative to a fixed object**, at a fixed radial sample density
  (`n_radial / R` constant): 0.244 to 0.140 from `R=40` to `R=80` at `n_angular=15`, and 2.16 to 1.43 to 0.928
  from `R=10` to `R=20` to `R=40` at `n_angular=63`: a factor of about 0.6 per doubling of `R`, more
  slowly than $1/R$. For $m \gg |n|$, McMahon's expansion
  $j_{nm} \approx (m + n/2 - 1/4)\pi$ gives a frequency-side radius offset between spoke $q$ and
  harmonic $n$ of about $(|n| - |q|)\,\pi / (2R)$, independent of $N_1$, which explains the direction of
  this trend but not its rate.
- **Per-spoke interpolation is the bridge from a uniform polar array.** The spline column reproduces the
  on-grid column to the digits shown in every case. `pypft.grid.resample_uniform_polar` is this
  interpolation, on uniform radii in `pypft.geometry.cartesian_to_polar`'s convention ($k R / n$,
  $k = 0, \dots, n - 1$): on the `n_radial=382, n_angular=15, R=40` grid it reproduces the on-grid
  relative L2 error to three digits (0.243 centered, 0.244 off-center), and on `n_radial=64,
  n_angular=15, R=10` its `forward_pft` result is within $1.1 \times 10^{-5}$ (relative L2) of the
  on-grid one. Fed as-is, a uniform polar array is read as if sample
  $(p, k)$ sat at $r_{pk}$, i.e. as a function warped differently along each spoke. For the centered
  Gaussian at `n_angular=15` that happens to score *better* (0.071), only because identical radii on
  every spoke avoid the spurious harmonics of a circularly symmetric function; at `n_angular=63` it
  scores worse (1.73 against 1.07). Even the uniform radii matter: with `cartesian_to_polar`'s radii,
  which start at the center, the same centered case scores 0.196 as-is, and the off-center one 0.249.
  None of this is a property of the data.
- **The ring route shows the error is the identification, not the conventions.** It computes harmonic
  $n$ from true rings at $r = r_{nk}$ (a 4096-point angular quadrature), applies the order-$|n|$ DHT,
  and evaluates the result off-grid at each output spoke's own $\rho_{qm}$ through the DHT's
  Fourier-Bessel expansion,
  $F_n(\rho) \approx 2\pi i^{-n} s_n \frac{2R^2}{j_{nN}^2} \sum_k \frac{f_n(r_{nk}) J_{|n|}(\rho\, r_{nk})}{J_{|n|+1}^2(j_{|n|k})}$
  ($s_n$ the negative-order sign), then sums $F_n(\rho_{qm}) e^{in\psi_q}$ over $n$. Every piece is
  PyPFT's own DHT/DFT math; only the spoke-to-harmonic identification differs, and it reaches rounding
  error wherever `n_angular` holds enough harmonics for the function (the `n_angular=15` off-center rows
  stop at 2e-2 from angular truncation). It is not exactly invertible and is not part of the package.

**Why `E_avg` alone is misleading.** `E_avg` averages a logarithm over all samples, so it is dominated by
the many samples where both the computed and the exact transform are essentially zero. It reports
−98.20 dB for a case whose relative L2 error is 2.16, an output that is no approximation at all. It
reproduces Yao & Baddour's published figures and nothing more, which is why `pypft.grid.check_adequacy`
is fitted on the relative L2 error instead (next section). Accuracy statements in this package report
the relative L2 error alongside `E_avg`.

**The three-part answer** to "can the PFT be applied to a uniformly polar-sampled image?":

1. As a discrete transform, `forward_pft`/`inverse_pft` apply to any `(n_radial, n_angular)` array,
   uniformly sampled or not: they invert each other exactly, and every discrete rule (orthogonality,
   shift, convolution, Parseval) holds. Part II states that the transforms can be applied to any matrix.
2. As an approximation of the continuous 2-D Fourier transform, the input must be samples at
   $(r_{pk}, \theta_p)$, and the output approximates samples at $(\rho_{qm}, \psi_q)$. A uniform polar
   image must be interpolated along each spoke onto `PolarGrid.r` first
   (`pypft.grid.resample_uniform_polar`), never fed as-is.
3. Even on the grid, the approximation carries the identification error above. It is small only when the
   function's content varies slowly across the per-spoke radius offsets; the relative L2 error, not
   `E_avg`, is the honest measure of it.

## Grid: `check_adequacy` is fitted on the relative L2 error

`pypft.grid.check_adequacy` predicts the forward transform's relative L2 error from
`(n_angular, n_radial)` alone, with a log-log least-squares fit
$\log_2 e = c_0 + c_1 \log_2 n_\text{angular} + c_2 \log_2 n_\text{radial}$, and warns when the
prediction exceeds a threshold. The measured quantity is the relative L2 error of `forward_pft` of the
centered Gaussian $e^{-r^2}$ sampled on the grid, against $\pi e^{-\rho^2/4}$ on the grid's own
$(\rho_{qm}, \psi_q)$, at `R=40`:

| `n_angular` | `n_radial` = 383 | `n_radial` = 767 | `n_radial` = 1535 |
| ----------- | ---------------- | ---------------- | ----------------- |
| 15          | 0.2422           | 0.1261           | 0.0784            |
| 32          | 0.4343           | 0.2477           | 0.1619            |
| 64          | 0.6439           | 0.4265           | 0.3052            |

The fit is $c_0 = 0.506$, $c_1 = 0.818$, $c_2 = -0.687$: each doubling of `n_angular` multiplies the
error by about 1.76, and each doubling of `n_radial` divides it by about 1.6. Its largest residual is
0.152 in $\log_2$, a factor of 1.11. Smaller grids leave the log-linear regime as the error approaches 1
(0.67 at `n_radial=95, n_angular=15`, 0.92 at `n_radial=95, n_angular=64`), so they are not fitted.

The threshold is 0.25: Yao & Baddour's own worked example, `n_radial=382, n_angular=15, R=40`, measures
0.243 (predicted 0.219), so a grid is reported when it is predicted to approximate the continuous
transform worse than that reference grid does. `n_radial=383, n_angular=64` (predicted 0.72, and
measured to give a positive `E_max`) warns; the suggested `n_radial` solves the fit for the threshold.

For the centered Gaussian, the error falls with `n_radial`: its exact transform has harmonic 0 alone,
so only the space-side radius offsets between spokes contribute, and they shrink as the grid gets
denser. The fit cannot see what the check has no input for. An off-center function also carries the
frequency-side offset of about $(|n| - |q|)\,\pi / (2R)$, which does not shrink with `n_radial` (0.244
to 0.239 from doubling it at `R=40`, previous section), and the check does not know `R` relative to the
object. A grid it accepts can therefore still give a large error; silence is necessary, not sufficient.

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

## Notebooks: citations link to raw-HTML anchors in a References cell

A tutorial notebook cites a source as `[[1]](#ref-1) (Eq. 41)` and ends with a Markdown cell headed
`## References` whose entries start `1. <a id="ref-1"></a>`. The format needs no code cell to render, so it
reads the same in JupyterLab, on GitHub, and in the Sphinx build, executed or not. JupyterLab stores each
anchor's `id` as `data-jupyter-id` and resolves `#ref-<k>` links against it, the same way it handles its
own heading anchors, so the links jump to their entries there. GitHub's renderer sanitizes `id`s, so there
the links may not jump; the anchor is an empty element either way, so it adds no visible text.

In the Sphinx build, MyST turns every `[text](#target)` link into a cross-reference and resolves it
against headings and explicit targets only. A raw-HTML anchor is neither, so each citation raises a
`myst.xref_missing` warning, which `sphinx-build -W` turns into an error. The link itself still works in
the built page: MyST falls back to a plain `href="#ref-1"`, and the raw `<a id="ref-1">` is passed
through to the same page. `docs/conf.py` therefore silences exactly the `ref-<k>` targets via
`nitpick_ignore_regex`, which MyST's resolver consults before warning, and leaves every other
cross-reference warning intact. The two alternatives MyST resolves natively both render badly
elsewhere: a `(ref-1)=` target line shows up as literal text outside MyST, and heading-per-entry anchors
are slugged differently by each viewer. Since the ignore disables Sphinx's own check for these links,
`tests/test_notebook_citations.py` checks that every `#ref-<k>` link has a matching anchor in its own
notebook.
