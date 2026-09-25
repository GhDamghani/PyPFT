# PyPFT

PyPFT is a Python toolkit for the polar Fourier transform — an angular DFT around a discrete Hankel transform — with applications such as reconstructing radially-sampled MR images from k-space.

Given frequency-domain samples in polar coordinates $F(\rho, \varphi)$, the spatial-domain image in polar coordinates $f(r, \theta)$ follows from one angular FFT, one discrete Hankel transform per harmonic, and one inverse angular FFT:

$F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)$

## Installation

```bash
pip install pypft
```

## Quickstart

```python
import numpy as np
import pypft

# The transform's own polar sampling grid
grid = pypft.PolarGrid(n_radial=383, n_angular=15, R=40.0)

# A Gaussian sampled on that grid, in PyPFT's (radial, angular) layout
f = np.exp(-(grid.r.T**2))

# Forward and inverse polar Fourier transform
F = pypft.forward_pft(f, grid)
f_reconstructed = pypft.inverse_pft(F, grid)
```

## Documentation

- [`docs/`](docs/) — the Sphinx documentation: quickstart, user guide, tutorials, API reference, and contributor setup.
- [`notebooks/`](notebooks/) — the tutorial notebook sequence, starting at `00_installation_and_quickstart.ipynb`.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to report issues, set up a development environment, and open a pull request.

## Citing

PyPFT's discrete Hankel transform and polar Fourier transform follow:

- Baddour, N. (2019). The Discrete Hankel Transform. In *Fourier Transforms - Century of Digitalization and Increasing Expectations*. IntechOpen. <https://doi.org/10.5772/intechopen.84399>
- Baddour, N. (2019). Discrete Two-Dimensional Fourier Transform in Polar Coordinates Part I: Theory and Operational Rules. *Mathematics*, 7(8), 698. <https://doi.org/10.3390/math7080698>
- Yao, X., & Baddour, N. (2020). Discrete Two-Dimensional Fourier Transform in Polar Coordinates Part II: Numerical Computation and Approximation of the Continuous Transform. *PeerJ Computer Science*, 6, e257. <https://doi.org/10.7717/peerj-cs.257>

The MR reconstruction application is based on:

- Golshani, S., & Nasiraei‐Moghaddam, A. (2017). Efficient radial tagging CMR exam: A coherent k‐space reading and image reconstruction approach. *Magnetic Resonance in Medicine*, 77(4), 1459-1472. <https://doi.org/10.1002/mrm.26219>

`pypft.Reference`, `pypft.cite`, and `pypft.bibliography` render the same sources programmatically.

## License

BSD-3-Clause — see [`LICENSE`](LICENSE). Third-party material shipped with the test suite is attributed in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
