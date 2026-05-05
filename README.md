# PyPFT
Polar Fourier Transform for Reconstruction of Polar MR images in Python using Numpy

Assuming you have raw kspace data in polar coordinates $F\left( {\rho ,\varphi } \right)$, if you want to reconstruct the image in the spatial domain with polar coordinates $f\left( {r,\theta } \right)$, you can follow these steps:

$F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)$

Meaning that you need to take two FFTs and one Hankel Transform. FFT is already implemented in `numpy.fft.fft`. However, Hankel transform is not implemented. Also, in general, there is no package to handle this type of data natively in Python. This package serves as a toolkit to reconstruct polar MR images using PFT and handle the images after that.

Based on:

    Golshani, S., & Nasiraei‐Moghaddam, A. (2017). Efficient radial tagging CMR exam: A coherent k‐space reading and image reconstruction approach. Magnetic resonance in medicine, 77(4), 1459-1472. https://doi.org/10.1002/mrm.26219

## Dependencies

<!-- dependencies:start -->
```text
numpy>=2.1
matplotlib>=2.0
joblib>=1.5
opencv-python>=4.12
platformdirs>=4.4
scipy>=1.16
```
<!-- dependencies:end -->

## Installing

Use the cross-platform installer script from the repository root:

```bash
python3.14 scripts/install_env.py
```

On Windows, if `python3.14` is not on `PATH`, use:

```powershell
py -3.14 scripts/install_env.py
```

The installer checks that it is running with Python 3.14, tells you how to install Python 3.14 if it is missing, creates `.venv`, upgrades `pip`, syncs `pyproject.toml` and `README.md` from `requirements/runtime.txt`, installs the runtime dependencies, and installs PyPFT in editable mode.

To include local development tools such as `invoke`, `build`, and `sphinx`, run:

```bash
python3.14 scripts/install_env.py --dev
```

To update `pyproject.toml` and `README.md` after changing `requirements/runtime.txt`, run:

```bash
python3.14 scripts/sync_dependencies.py
```

For local maintenance commands, install the dev extras and use `invoke`:

```bash
python3.14 scripts/install_env.py --dev
inv --list
```

Available tasks include:

- `inv install --dev` to create the environment with dev tools installed
- `inv sync-deps` to sync dependency metadata
- `inv backfill-dist --dry-run` to preview missing distribution artifacts
- `inv docs` to build the Sphinx docs into `docs/build/html`

You can run a sample test with this command:

```bash
python -m pypft test
```

## Release automation

Merged pull requests into `main` can trigger an automated release based on the source branch prefix.

- `major/*` forces a major release.
- `minor/*` forces a minor release.
- `patch/*` forces a patch release.

The GitHub Actions workflow at `.github/workflows/release.yml` runs after a qualifying pull request is merged into `main`, creates the release with `python-semantic-release`, publishes release artifacts to GitHub, and publishes the package to PyPI when a release is produced.
