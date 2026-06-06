# PyPFT

PyPFT is a Python package for exact discrete Fourier transforms in polar coordinates. The public API is centered on a single facade object:

```python
from pypft import PyPFT

pft = PyPFT()
transformed = pft.forward(image)
reconstructed = pft.backward(transformed)
```

The current repository state provides the installable package, the `PyPFT` facade, modular FFT and operator layers, a single legacy-style direct DHT implementation, and a Typer + Rich CLI for traced transforms, artifact replay, and validation-oriented reporting. The compatibility key remains `naive`, but it now runs a reference radial transform derived from the pre-0.1 codebase instead of an identity passthrough.

## Current design

- Default input contract: one 2D `complex128` array with shape `(n_r, n_theta)`.
- Optional batched contract: one leading batch axis with shape `(batch, n_r, n_theta)` when `enable_batching=True`.
- Angular transforms: backend-dispatched FFT wrappers using SciPy on CPU and optional CuPy on GPU.
- Radial transforms: DHT selected by string key from a registry; the compatibility key is `naive`, and the same legacy-style direct implementation is reused by forward and backward plans.
- Backend support: `backend="cpu"` is the default; `backend="gpu"` is available when the optional CuPy extra is installed.
- Operator layer: separate forward and backward plans keep the facade small and the stages independently testable.
- Trace model: traced runs expose the canonical public stages `spatial_samples`, `angular_dft`, `radial_dht`, `angular_idft`, and `frequency_samples`.
- Artifact model: transform runs write `output.npy`, `trace.json`, and `manifest.json`; optional stage arrays and figures are stored under `stages/` and `figures/` only when requested.
- Visualization and reporting: the CLI supports static Matplotlib renders, artifact replay, trace inspection, comparison figures, and roundtrip validation reports. Matplotlib is the only shipped renderer backend, and no TUI is shipped yet.

## Reference materials

Authoritative papers and derivations are stored under `.local_files/sources`. The current scaffold freezes structural conventions such as axis order and dtype policy while the new reference DHT is validated. Mathematical normalization, phase conventions, and final grid sampling semantics still need to be hardened beyond the legacy baseline.

## Installation

PyPFT currently targets Python 3.14.

To create a local development environment:

```bash
python scripts/install_dev.py
```

That script creates `.venv`, upgrades `pip`, and installs the package in editable mode with development, documentation, and benchmark extras.

For a minimal editable install:

```bash
python -m pip install -e .
```

For static figure rendering, trace replay figures, and validation comparison plots:

```bash
python -m pip install -e .[viz]
```

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
dispatch only. The compatibility `naive` DHT still uses a CPU-side
reference kernel and converts the result back to the active array backend, so
this does not yet represent real GPU-accelerated radial transform math.

The package also installs a `pypft` console script.

## CLI workflows

Transform commands require sidecar metadata in `.pypft.json` format. `transform forward` expects metadata with `domain: "spatial"`; `transform backward` expects `domain: "frequency"`. All figure-producing commands support `--gamma` and `--complex-view`; the historical alias `angular` is still accepted for `phase`.

Run a traced forward transform and save all figures plus intermediate arrays:

```bash
pypft transform forward input.npy \
    --output-dir artifacts \
    --save-all-views \
    --save-stage-arrays
```

Replay or inspect saved artifacts:

```bash
pypft visualize trace artifacts/manifest.json --output-dir replayed
pypft visualize trace artifacts/trace.json --output-dir replayed-from-arrays
pypft inspect trace artifacts/manifest.json
```

Render a standalone field array without rerunning the transform:

```bash
pypft visualize field stage.npy \
    --field-kind radial_spectrum \
    --output-dir figures
```

Run validation-oriented reporting:

```bash
pypft validate roundtrip input.npy --output-dir validation
pypft validate compare reference.npy candidate.npy --output-dir comparison
```

`validate roundtrip` writes `forward.npy`, `reconstruction.npy`, `roundtrip-report.json`, and comparison figures for the input versus reconstruction. `validate compare` writes `comparison-report.json` plus comparison figures for two saved arrays. Both validation commands report max absolute error, RMSE, relative $L_2$ error, and pass/fail status using practical defaults `--atol 1e-10` and `--rtol 1e-10`.

## Artifact layout

Transform manifests and trace documents use relative artifact paths rooted at the output directory. Figure names follow the `<stage-stem>.<view>.png` convention.

A typical transform run can produce:

```text
artifacts/
        output.npy
        trace.json
        manifest.json
        stages/
                01_spatial_samples.npy
                02_angular_dft.npy
                ...
        figures/
                01_spatial_samples.magnitude.png
                01_spatial_samples.phase.png
                ...
```

`visualize trace <manifest.json>` can replay the transform from the saved manifest even when stage arrays were not saved. `visualize trace <trace.json>` renders directly from the saved trace document and therefore requires `array_path` entries, which are only present when `--save-stage-arrays` was enabled during the original run.

## Development checks

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
    cli/
    config.py
    backends/
    core/
    dft/
    dht/
    fields/
    grids/
    idft/
    io/
    operators/
    tracing/
    validation/
    visualization/
    workflows/
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
