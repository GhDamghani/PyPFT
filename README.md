# PyPFT

PyPFT is a Python package for exact discrete Fourier transforms in polar coordinates. The public API is centered on a single facade object:

```python
from pypft import PyPFT

pft = PyPFT()
transformed = pft.forward(image)
reconstructed = pft.backward(transformed)
```

The current repository state provides the installable package scaffold, the `PyPFT` facade, modular FFT and operator layers, and a single mock DHT implementation. The current DHT key is `mock-mirror`, which simply returns its input so the existing forward and backward pipelines behave like identity transforms while the real radial math is still under development.

## Current design

- Default input contract: one 2D `complex128` array with shape `(n_r, n_theta)`.
- Optional batched contract: one leading batch axis with shape `(batch, n_r, n_theta)` when `enable_batching=True`.
- Angular transforms: backend-dispatched FFT wrappers using SciPy on CPU and optional CuPy on GPU.
- Radial transforms: DHT selected by string key from a registry; the current key is `mock-mirror`, and the same implementation is reused by forward and backward plans.
- Backend support: `backend="cpu"` is the default; `backend="gpu"` is available when the optional CuPy extra is installed.
- Operator layer: separate forward and backward plans keep the facade small and the stages independently testable.

## Reference materials

Authoritative papers and derivations are stored under `.local_files/sources`. The current scaffold freezes only structural conventions such as axis order and dtype policy. Mathematical normalization, phase conventions, and grid sampling semantics remain pending until the DHT implementation work begins.

## Installation

PyPFT currently targets Python 3.14.

To create a local development environment:

```bash
python scripts/install_dev.py
```

That script creates `.venv`, upgrades `pip`, and installs the package in editable mode with development, documentation, and benchmark extras.

To include the optional CuPy GPU backend during environment setup:

```bash
python scripts/install_dev.py --gpu
```

You can also install manually:

```bash
python -m pip install -e .[dev,docs,bench]
```

For optional GPU support on CUDA 12.x:

```bash
python -m pip install -e .[dev,docs,bench,gpu]
```

Then select the GPU path explicitly:

```python
from pypft import PyPFT

pft = PyPFT(backend="gpu")
```

The current GPU milestone accelerates array conversion and angular FFT/IFFT
dispatch only. The current DHT implementation is still `mock-mirror`, so this
does not yet represent real GPU-accelerated radial transform math.

## Validation

Run the test suite with:

```bash
python -m pytest
```

Build the documentation with:

```bash
python -m sphinx -b html docs docs/build/html
```

Build source and wheel distributions with:

```bash
python -m build
```

## Repository layout

```text
src/pypft/
    api.py
    config.py
    backends/
    core/
    dft/
    dht/
    grids/
    idft/
    operators/
    validation/
    visualization/
tests/
docs/
scripts/
benchmarks/
notebooks/
```

## GitHub and publishing

The repository now includes GitHub Actions workflows for:

- test and package-build verification on push and pull request
- documentation builds on push and pull request
- merge-triggered version and changelog updates for `major/*`, `minor/*`, and `patch/*` branches merged into `main`
- tagged releases that build distributions, publish a GitHub Release, and publish to PyPI via Trusted Publishing

When a release-managed pull request is merged into `main`, the version workflow reads the branch prefix to choose the semantic version bump and uses the merge commit description body as the new changelog entry body.

Before enabling PyPI publishing, configure the PyPI Trusted Publisher for this repository and confirm that the release tags match the package version in `pyproject.toml`. 
