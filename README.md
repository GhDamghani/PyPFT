# PyPFT

Polar Fourier Transform for Reconstruction of Polar MR images in Python using Numpy

Assuming you have raw kspace data in polar coordinates $F\left( {\rho ,\varphi } \right)$, if you want to reconstruct the image in the spatial domain with polar coordinates $f\left( {r,\theta } \right)$, you can follow these steps:

$F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)$

Meaning that you need to take two FFTs and one Hankel Transform. FFT is already implemented in `numpy.fft.fft`. However, hankel transform is not implemented. Also, in general, there is no package to handle this type of data natively in Python. This package serves as a toolkit to reconstruct polar MR images using PFT and handle the images after that.

Based on:

    Golshani, S., & Nasiraei‐Moghaddam, A. (2017). Efficient radial tagging CMR exam: A coherent k‐space reading and image reconstruction approach. Magnetic resonance in medicine, 77(4), 1459-1472. https://doi.org/10.1002/mrm.26219

## User Guide

### Installation

```powershell
uv add pypft
```

### The discrete Hankel transform

```python
import numpy as np
import pypft

n, R = 0, 8.0
r, rho = pypft.sample_points(n, size=64, R=R)

f = np.exp(-(r**2) / 2)
F = pypft.hankel_transform(f, n, R)
f_reconstructed = pypft.inverse_hankel_transform(F, n, R)
```

### The angular discrete Fourier transform

`pypft.dft.angular_dft`/`pypft.dft.inverse_angular_dft` compute the centered
angular DFT/IDFT -- the `FFT_phi`/`IFFT_theta` steps at the top of this
file's chain, kept centered the same way every other PyPFT array is (index
`n_angular // 2` holds harmonic `0`). This is a lower-level building block
for the eventual PFT/IPFT pipeline rather than a typical end-user entry
point, so it is not re-exported from the top-level `pypft` package; see the
API reference for the full signature, `pypft.dft.harmonics` (the centered
harmonic-index range for a given angular sample count, valid for either an
odd or an even count), and `pypft.dft.AngularParity`:

```python
import numpy as np
from pypft.dft import angular_dft, inverse_angular_dft

x = np.random.default_rng(0).standard_normal(16)
X = angular_dft(x)
x_reconstructed = inverse_angular_dft(X)
```

### Cartesian and polar images

`pypft.cartesian_to_polar`/`pypft.polar_to_cartesian` resample an ordinary
image onto (and back off of) a *uniform* polar grid. This is not the
transform's own (order-dependent, non-uniform) sampling grid -- it is the
natural first illustration of what "polar" means for an image:

```python
polar = pypft.cartesian_to_polar(image, n_radial=128, n_angular=96)
reconstructed = pypft.polar_to_cartesian(polar, height, width)
```

The returned array follows PyPFT's own `(radial, angular)` axis layout
(`pypft.Axis`), with a centered angular axis: index `n_angular // 2` holds
angle `0`.

### The transform's own sampling grid

`pypft.PolarGrid` is the discrete Hankel transform's *actual* sampling grid --
order-dependent and non-uniform, unlike `cartesian_to_polar`'s uniform one above.
Every angular row has its own radial sample positions, tied to the zeros of a
Bessel function of that row's harmonic order:

```python
grid = pypft.PolarGrid(n_radial=383, n_angular=15, R=40.0)
grid.r      # (n_angular, n_radial) space-domain radii
grid.theta  # (n_angular,) centered angles, shared by both domains
```

`pypft.sample_cartesian` resamples an ordinary image directly onto a grid's own
points -- the production sampler, as opposed to `cartesian_to_polar`'s
illustrative uniform one:

```python
polar = pypft.sample_cartesian(image, grid)
```

`pypft.check_adequacy`/`pypft.check_nyquist_adequacy` warn (never raise) when a
grid's `n_radial` is too small for its `n_angular`, or violates the discrete
Hankel transform's own Nyquist condition, respectively -- both are easy mistakes
to make silently, since neither failure mode raises an error on its own:

```python
pypft.check_adequacy(grid)                      # silent for this grid
pypft.check_adequacy(pypft.PolarGrid(383, 64, 40.0))  # warns: n_radial too small
```

### The full PFT/IPFT pipeline

`pypft.forward_pft`/`pypft.inverse_pft` chain the angular DFT/IDFT with a
per-harmonic, `R`-scaled discrete Hankel transform -- the complete chain at
the top of this file. Both take a `pypft.PolarGrid` and a `(n_radial,
n_angular)` array on PyPFT's own layout (`pypft.Axis`) -- **not**
`PolarGrid.r`'s/`pypft.sample_cartesian`'s own `(n_angular, n_radial)`
layout, so transpose a `sample_cartesian` result first:

```python
import numpy as np
import pypft

grid = pypft.PolarGrid(n_radial=382, n_angular=15, R=40.0)
f = np.exp(-(grid.r.T**2))            # a radially symmetric Gaussian, on grid.r.T
F = pypft.forward_pft(f, grid)        # the frequency-domain samples F(rho, phi)
f_reconstructed = pypft.inverse_pft(F, grid)
```

### Citing a result

`pypft.Reference`/`pypft.cite`/`pypft.bibliography` render the scientific
sources behind PyPFT's math:

```python
pypft.cite(pypft.Reference.BADDOUR_2019_DHT)
pypft.bibliography(pypft.Reference.BADDOUR_2019_DHT)
```

See `notebooks/00_installation_and_quickstart.ipynb`,
`notebooks/01_polar_and_cartesian_images.ipynb`,
`notebooks/02_sampling_grids.ipynb`, and `notebooks/03_pft_and_ipft.ipynb`
for the full walkthrough.

## Developer Guide

### Installing

- Make sure you have [`uv`](https://docs.astral.sh/uv/) installed.

- Clone the repository.

- Run this command:

  ```powershell
  uv sync
  ```

- Install the recommended VS Code extensions.

**Important note**: Before start working on a branch, always run `uv sync` first to make sure your environment is update to the `pyproject.toml`.

<!-- This is for the linter to accept the inline HTML for GitHub-Flavored Markdown -->

<!-- markdownlint-disable MD033 -->

<details>
<summary>Notes on developer dependencies and recommended extensions</summary>

#### Dev Dependencies

- `black`: Python formatter.
- `cython`: Optimizing static compiler.
- `flake8-rst-docstrings`: Docstring reStructuredText (RST) validator for flake8.
- `ipdb`: IPython version of the `pdb` debugger.
- `ipython`: Enhanced interactive Python shell.
- `isort`: Sorts Python imports
- `matplotlib`: Visualization tool. Useful for quick debugging related to signals.
- `notebook`: notebook environment for interactive computing. Could also be useful later for example scripts.
- `pyment`: Generate/convert automatically the docstrings from code signature. Useful to make sure all docstrings are reST, and no method is left without a docstring.
- `pytest`: Unit testing environment.
  - `pytest-cov`: adds coverages stats.
  - `pytest-vulture`: Finds dead code.
- `setuptools`: Python build backend.
- `tqdm`: CLI Progress-bar.
- `wheel`: Required for wheel files - for packaging.

#### Extension recommendations

- Python `ms-python.python`: Python language support.
- Black Formatter `ms-python.black-formatter`: Black formatter.
- Flake8 `ms-python.flake8`: Linting support.
- Pylance `ms-python.vscode-pylance`: Language server.
- Python Debugger `ms-python.debugpy`: Debugger.
- Python Indent `KevinRose.vsc-python-indent`: Correct Python indentation.
- Code Spell Checker `streetsidesoftware.code-spell-checker`: Spellchecker.
- isort `ms-python.isort`: Sorts Python Imports.
- autoDocstring - Python Docstring Generator `njpwerner.autodocstring`: Make template for docstring (adjusted to reST).
- Markdown Table Prettifier `darkriszty.markdown-table-prettify`: Transforms markdown tables to be more readable.
- GitHub Markdown Preview `bierner.github-markdown-preview`: Preview Markdown files based on GitHub-Flavored Markdown.
- Rewrap Revived `dnut.rewrap-revived`: Hard word wrapping for comments and other text.

</details>
<!-- markdownlint-enable MD033 -->

### Development Convention

#### Constant Extraction

Practice extracting constant as much as possible. If it is only on the module-level, put it at the top of module. If not, find a right level of sharing and put it there.

#### Errors and Warnings

All errors should be handled with Exceptions. This means that when calling a method that could raise an exception, you should use `try` statements with `except` clauses. If the generic exceptions are not enough to handle them, you should make new exceptions by [inheriting](https://docs.python.org/3/library/exceptions.html#inheriting-from-built-in-exceptions) from their generic type. For instance, if a function validates the value of two variables `val1` and `val2`, and the software should behave differently, Inherit `ValueError` to make two new exceptions `Val1ValueError(ValueError)` and `Val2ValueError(ValueError)` and use them.

Also, use `warnings` for warnings.

<!-- markdownlint-disable MD033-->

<details>
<summary>Expand to see an example implementation</summary>

```python
import warnings


class Val1ValueError(ValueError):
    """Raised when val1 fails validation."""

    def __init__(self, message="val1 is invalid"):
        super().__init__(f"{message}")


class Val2ValueError(ValueError):
    """Raised when val2 fails validation."""

    def __init__(self, message="val2 is invalid"):
        super().__init__(f"{message}")


def validate(val1: int, val2: int) -> None:
    if val1 < 0:
        raise Val1ValueError(f"val1 must be non-negative, got {val1!r}")
    if val2 == 0:
        raise Val2ValueError(f"val2 must not be zero, got {val2!r}")
    if val1 > 1000:
        warnings.warn(f"val1={val1} is unusually large", stacklevel=2)


def process(val1: int, val2: int) -> float:
    DEFAULT_VAL1 = 5
    try:
        validate(val1, val2)
    except Val1ValueError as e:
        warnings.warn(
            f"val1={val1} is an invalid value, substituting with {DEFAULT_VAL1}",
            stacklevel=2,
        )
        val1 = DEFAULT_VAL1
    except Val2ValueError as e:
        raise Val2ValueError(
            f"Cannot process values because val2 is invalid: {e}"
        ) from e
    return val1 / val2


# Usage

# Emits a UserWarning (val1 too large), still works
process(2000, 5)

# Emits a UserWarning (val1 is invalid), but since there is default fallback, it still
# works.
process(-1, 5)

# Raises Val2ValueError
process(10, 0)
```

</details>
<!-- markdownlint-enable MD033 -->

#### Annotations and Validators

All function signatures must use type annotations, and validate their input. Usage of validators for function outputs is recommended.

Validators for all types (**except locally-defined types**) are located in `src/artinis_poly6/lib/utils/validators.py`. Validators for locally-defined types are **Defined in their own class**. Read the file docstring for more information.

### Styling Conventions

The style conventions of this project is based on [PEP 8](https://peps.python.org/pep-0008/). However, there is one exception:

- Maximum Line Length: It's `88` instead of `79`, as recommended by [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#line-length). It's also not forced on comments, docstrings, and strings. However, we should try to break them. The "Rewrap Revived" extension is helpful for comments and docstrings.

#### Documentation Strings

We follow [PEP 287](https://peps.python.org/pep-0287/) and write in [reStructuredText markup](https://docutils.sourceforge.io/rst.html).

Recommended extensions are helpful to make sure we adhere to these conventions.

#### Spellings

Use `streetsidesoftware.code-spell-checker` for spellchecking. It's set to American English.

#### Code sectioning with comments

Use `# ` followed by the sectioning-character, filling the line length to max. <!-- markdownlint-disable-line MD001 MD038 -->

- Section: `=`
- Subsection: `-`
- Indented section: `*`
- Indented subsection: `.`

Example:

```python
# ========================================================================================
# Section
# ========================================================================================

    # ************************************************************************************
    # Indented section
    # ************************************************************************************

    # ....................................................................................
    # Indented subsection
    # ....................................................................................


# ----------------------------------------------------------------------------------------
# Subsection
# ----------------------------------------------------------------------------------------

    # ************************************************************************************
    # Indented section
    # ************************************************************************************

    # ....................................................................................
    # Indented subsection
    # ....................................................................................
```

There are VS Code snippets define for it in `.vscode\helpers.code-snippets`. Their shortcut (prefix) are as followed:

- Section: `@ section`
- Subsection: `@ subsection`
- Indented section: `@ isection`
- Indented subsection: `@ isubsection`
