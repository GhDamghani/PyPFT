# Third-Party Notices

This project's own source code is licensed under the terms in `LICENSE` (BSD
3-Clause). This file lists third-party assets bundled in the repository that
carry their own license/attribution terms, separate from that license.

## `tests/samples/hedge_maze.tif`

A 576x39 grayscale raster (`pypft`'s own `(n_radial, n_angular)` layout, not
the original square artwork) derived from a stock icon licensed from
Vecteezy (Eezy Inc.), downloaded as
`vecteezy_hedge-maze-icon-style_9215823.eps` (asset id `9215823`) under
Vecteezy's Free License. The Free License permits reuse but requires
attribution to the source rather than granting unrestricted reuse;
`scripts/make_test_image.py` rasterized the original artwork, resized it to
a 256x256 square, and sampled that square onto a
`pypft.PolarGrid(n_radial=576, n_angular=39, R=121.6)` via
`pypft.sample_cartesian` to produce this file -- used as the sample
`Domain.SPACE_POLAR` signal in `notebooks/07_visualization.ipynb`. This grid
size keeps the transform's own per-harmonic kernel stack under ~200MB while
`pypft.check_adequacy` still raises no warning.

Attribution: **Vecteezy.com** — <https://www.vecteezy.com>

The Vecteezy Terms of Use (<https://www.vecteezy.com/terms-of-use>) govern
use of this asset and supersede any summary given here.
