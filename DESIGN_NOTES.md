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
  stop at 2e-2 from angular truncation). It is not exactly invertible. `pypft.ring` implements it as
  public API; see "PFT: the exact path and the ring-consistent route" below.

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

## PFT: the exact path and the ring-consistent route

PyPFT has two ways to transform polar data. They trade exact invertibility against accuracy as an
approximation of the continuous 2-D Fourier transform, so neither replaces the other.

- **The exact path** (`pypft.forward_pft`/`pypft.inverse_pft`, and every `pypft.BaseSignal` step) is
  Baddour's discrete PFT on `PolarGrid`: square and exactly invertible for any array, with every discrete
  rule (orthogonality, shift, convolution, Parseval) holding exactly. As an approximation of the
  continuous transform it carries the spoke-to-harmonic identification error of the grid section above.
  It is the default everywhere.
- **The ring-consistent route** (`pypft.ring`: `forward_pft_ring`/`inverse_pft_ring` and their pieces)
  avoids that identification. It is not a square matrix on one array, so it is not exactly invertible:
  its inverse is a mirror-image approximation, and the discrete rules hold only approximately.

**The route.** Forward, in three stages, with no new DHT or DFT kernel:

1. *True rings in.* Harmonic $n$ is computed from samples on rings at its own radii $r_{nk}$ (row $i$ of
   `PolarGrid.r` for `harmonics[i] == n`), by an angular quadrature around each ring, scaled to
   `pypft.dft.angular_dft`'s own scale. `sample_harmonics_cartesian` samples a Cartesian image at
   `n_quadrature` angles (default one per pixel of arc on the outermost ring); `sample_harmonics_uniform_polar`
   interpolates every spoke of uniform polar data along the radius (a cubic spline) and uses only the
   data's own spokes as the quadrature. Uniform polar data already has true rings, so no angle is
   interpolated. The result is a spatial-harmonic array: it enters the domain chain one step in.
2. *The existing per-harmonic DHT*, `pypft.transform.scaled_hankel`, gives $F_n(\rho_{nm})$.
3. *Each spoke's own radii out.* `evaluate_frequency` evaluates every $F_n$ at every output spoke's
   $\rho_{qm}$ through the DHT's Fourier-Bessel expansion, and sums
   $\frac{1}{N_2} \sum_n F_n(\rho_{qm}) e^{i n \psi_q}$ on each spoke. With $g = Y^{|n|N} x$ the
   dimensionless DHT of samples $x$ at the order-$|n|$ points, the function is
   $\frac{2}{j_{|n|N}} \sum_k g_k J_{|n|}(u\, j_{|n|k}) / J_{|n|+1}^2(j_{|n|k})$ at the scaled radius
   $u = \rho R / j_{|n|N}$ for frequency harmonics, and $u = r / R$ for spatial harmonics (the inverse
   DHT's Fourier-Bessel series, used by `evaluate_space`). At another order $q$'s grid radii $u$ is
   $j_{|q|m} / j_{|n|N}$ or $j_{|q|m} / j_{|q|N}$, free of $R$; at $q = n$ both matrices are
   $Y^{|n|N}$ itself, so the samples' own points are reproduced. The two sides differ only in that
   denominator. The Bessel values come from direct `scipy.special.jv` calls (see "DHT: the kernel's
   Bessel values must be computed directly" above). The harmonic sign and the $2\pi i^{\mp n}$ factors
   cancel out of the interpolation, so it needs no knowledge of `scaled_hankel`'s scaling.

The inverse mirrors it: every frequency spoke is interpolated along the radius (a cubic spline on its own
$\rho_{qm}$) onto every harmonic's $\rho_{nm}$, the quadrature over the $N_2$ spokes gives each harmonic,
the inverse `scaled_hankel` follows, and `evaluate_space` evaluates the result at every spatial spoke's
$r_{pk}$.

**Measurements.** Gaussians $e^{-|\mathbf{x} - \mathbf{x}_0|^2}$ against their exact transforms, as in
the grid section; relative L2 errors throughout. "Uniform" samples `n_radial` uniform radii in
`pypft.geometry.cartesian_to_polar`'s convention ($k R / n$) on the grid's own spokes, and feeds
`forward_pft_ring` with only those spokes as the quadrature. "Round trip" is `inverse_pft_ring` of the
uniform route's result against the function on the grid's spatial points; the exact path's round trip is
at rounding error ($\le 3 \times 10^{-13}$) in every row.

| `n_radial` | `n_angular` | `R` | $\mathbf{x}_0$ | exact path | ring, 4096-angle quadrature | ring, uniform | ring round trip |
| ---------- | ----------- | --- | -------------- | ---------- | --------------------------- | ------------- | --------------- |
| 64         | 15          | 10  | $(0, 0)$       | 0.320      |                             | 1.5e-5        | 1.1e-3          |
| 96         | 15          | 20  | $(1, -0.5)$    | 0.251      |                             | 4.7e-4        | 7.0e-4          |
| 382        | 15          | 40  | $(2, -1)$      | 0.244      | 1.95e-2                     | 2.9e-2        | 1.4e-4          |
| 764        | 15          | 80  | $(2, -1)$      | 0.140      |                             | 2.8e-2        | 3.8e-6          |
| 128        | 31          | 20  | $(1, -0.5)$    | 0.511      |                             | 7.9e-6        | 8.1e-3          |
| 200        | 63          | 10  | $(2, -1)$      | 2.16       | 4.6e-13                     | 1.1e-7        | 1.62            |
| 400        | 63          | 20  | $(2, -1)$      | 1.43       |                             | 9.0e-8        | 0.65            |

- **The forward route removes the identification error; its input stage sets what remains.** From
  analytic ring samples and a dense quadrature it reaches rounding error wherever `n_angular` holds the
  function's harmonics (4.6e-13); from uniform polar data the radial spline leaves 1e-7 to 1.5e-5; and
  angular truncation is what remains otherwise (the `n_angular=15` off-center rows). Using only the
  data's own spokes as the quadrature costs little: 2.9e-2 against 1.95e-2 on the paper's grid, where
  the 15 spokes alias harmonics beyond the grid's range, and nothing visible where the harmonics are
  resolved. Uniform polar data is therefore a practical input with no extra acquisition.
- **From a Cartesian image** the error is set by the bilinear pixel interpolation: a Gaussian of width
  6 pixels at $(8, -4)$ pixels, on `n_radial=64, n_angular=15, R=32` (pixels), measures 7.2e-3 (the
  exact path from `sample_cartesian`: 1.12); width 8 at $(16, -8)$ on `n_radial=128, n_angular=31,
  R=64` measures 3.7e-3 (exact path 1.57). The default quadrature and one of `n_angular` angles
  give errors within 10% of each other when the image is this smooth (7.2e-3 and 7.8e-3).
- **Discontinuous content.** A disk of radius 0.3 at $(0.2, -0.1)$, whose transform is
  $2\pi a J_1(\rho a)/\rho \, e^{-i \boldsymbol{\rho} \cdot \mathbf{x}_0}$, on `n_radial=192,
  n_angular=48, R=1` (the maze fixture's `R` and spoke count): exact path 2.70, ring route from four
  times as many uniform radii 0.067, from a 2048-angle quadrature 0.043. The maze fixture's two spectra
  (`maze_polar.tif` through `forward_pft`, and `scripts/make_maze.py`'s `sample_maze_rings` through the
  route) differ by 4.56 relative to the route's, consistent with the disk.

**The inverse is limited by the frequency grid's central gap.** Spoke $q$ has no sample below
$\rho_{q1} = j_{|q|1} / R$, and the spokes with the largest gaps are the ones far from angle 0, so near
the center of the frequency plane only a wedge of spokes around $\psi = 0$ holds samples. The inverse's
input stage extrapolates each spoke's spline across its own gap. That is accurate while the gap holds
little of the transform (large `R` relative to the object, few spokes: 1.4e-4 and 3.8e-6 above) and fails
when it holds much of it (`n_angular=63`, `R=10` or 20; the disk above, 4.06). Measured on the same
cases, setting the gap to zero or to the spoke's first sample is worse wherever extrapolation works
(7.8e-2 and 9.1e-3 on the paper's grid) and still fails where it does not (0.86 and 0.81 at
`n_angular=63, R=10`). Past a spoke's last sample, which harmonics of larger order reach by about
$(|n| - |q|)\pi / (2R)$, the value is taken as 0, the grid's own band-limit assumption: cubic
extrapolation over that many sample spacings diverges on broadband content (4.42 against 4.06 on the
disk) and changes nothing on the Gaussians. An exact inverse of the route would need the
$N_1 N_2 \times N_1 N_2$ evaluation operator inverted, which is out of reach at these sizes; the exact
path's `inverse_pft` remains the exact inverse.

**Cost, and why nothing is cached.** The output stage evaluates $(N_2 / 2 + 1)^2$ distinct order pairs
(harmonics $\pm n$ and spokes $\pm q$ share their Bessel orders), each an $N_1 \times N_1$ matrix of
`jv` values, about 0.15 s per matrix at `n_radial=576`. Measured per direction: 0.6 s at
`n_radial=382, n_angular=15`, 8 s at `n_radial=200, n_angular=63`, 25 s at `n_radial=400,
n_angular=63`, and 19 s at the maze's `n_radial=576, n_angular=48`, on 16 cores. `scipy.special.jv`
releases the GIL, so the output orders run on a thread pool: about 10 times faster than serially there
(85 s at `n_radial=200, n_angular=63`). A cache keyed on the grid, like `STACKED_KERNEL`'s, would hold
every pair's matrix: 1.7 GB at the maze's size, and 330 MB at `n_radial=200, n_angular=63`, against
`STACKED_KERNEL`'s single $N_2 N_1^2$ stack. A smaller LRU of single matrices would be swept through in
order on every call and never hit. Neither is used; the route is a single implementation, not a
`PFTImplementation` strategy, since it computes a different discretization rather than the same one
faster.

## Domains: one state word per coordinate group

A sample's coordinates fall into groups that the transform changes one at a time. A polar sample has two:
the radial coordinate, which the discrete Hankel transform takes from space (`SPATIAL`) to frequency
(`FREQUENCY`), and the angular coordinate, which the angular DFT takes from a physical angle (`ANGULAR`) to a
harmonic order (`HARMONIC`). The naming rule is that each coordinate system gets its own domain enum, and
a member's name is one state word per coordinate group, with no system prefix: `PolarDomain` has
`SPATIAL_ANGULAR`, `SPATIAL_HARMONIC`, `FREQUENCY_HARMONIC` and `FREQUENCY_ANGULAR`, in chain order. Naming
every group's state, rather than only the ones that differ from the starting point, keeps each name
meaningful on its own, and leaving the system out of the member keeps it in one place, the enum and the
signal class (`PolarSpatialAngularSignal`, ...). Step methods follow the member names (`to_spatial_harmonic`,
...), since the class a method is called on already names the system.

The rule is motivated by other coordinate systems with the same structure. A spherical sample would have a
radial group and an angular group whose harmonics are spherical harmonics, so a `SphericalDomain` would read
`SPATIAL_ANGULAR` → `SPATIAL_HARMONIC` → `FREQUENCY_HARMONIC` → `FREQUENCY_ANGULAR` without colliding with
`PolarDomain`. A cylindrical sample adds an axial group, which adds a third state word. Where two systems
must be told apart in one namespace, the system comes from the enum's class name: trace labels and the file
names `PFTTrace.save` writes are `step_<system>_<member>` (`step_polar_spatial_angular`).

Axes follow the same idea. `PolarAxis` names the two sample axes (`RADIAL = 0`, `ANGULAR = 1`) and nothing
else: the batch axis is not a coordinate of the sample but the axis after the sample axes, index
`POLAR_SAMPLE_NDIM` (2) for a polar sample, and it is named only by `DEFAULT_BATCH_AXIS = -1`. "3-D" never
means "a batch of 2-D samples", since a spherical or cylindrical sample is itself three-dimensional. Under
the naming layer, the steps stay axis-flexible: `pypft.dft.angular_dft`, `pypft.dht.hankel_transform` and
`pypft.transform.scaled_hankel` take their axes explicitly and accept any rank, so a layer with a different
layout applies the same steps on whichever axes it names. Only `forward_pft`/`inverse_pft` and the signal
classes fix the polar layout.

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
