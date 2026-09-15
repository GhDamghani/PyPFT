r"""Rasterize a source image and sample it onto a PolarGrid, for a test fixture.

Handles vector sources (``.eps``/``.ps``, via Pillow's Ghostscript-backed
``EpsImagePlugin``) transparently alongside ordinary raster formats
(``.png``/``.jpg``/``.tif``/...): either is rasterized/loaded, resized to a
square, then sampled via ``pypft.sample_cartesian`` onto a
``pypft.PolarGrid(n_radial, n_angular, R)`` -- the same order-dependent,
non-uniform sampling ``pypft.forward_pft``/``inverse_pft`` themselves consume,
unlike ``pypft.cartesian_to_polar``'s uniform illustration grid. The
intermediate square raster is never saved; only the final ``(n_radial,
n_angular)`` polar-sampled array is written to ``--output``, as an
uncompressed 8-bit grayscale TIFF -- ready to feed directly into
``pypft.SpacePolarSignal``/``pypft.forward_pft`` without re-sampling.

Usage:
    uv run python scripts/make_test_image.py --input <path> --output <path> \
        [--size 256] [--n-radial 1024] [--n-angular 127] [--r-fraction 0.95]

Prints ``pypft.check_adequacy``'s own verdict for the requested grid, since
choosing ``n_radial``/``n_angular`` so that check clears is the reason this
script takes them as separate parameters rather than hardcoding a grid.

Ghostscript is required only for a ``.eps``/``.ps`` source, and is resolved via
PATH (``gswin32c``/``gswin64c``/``gs``) first, falling back to the default
Windows install directory if PATH lookup fails (a silent installer does not
always update the current process's own PATH).

Expect this to take roughly two minutes at the ``--n-radial``/``--n-angular``
defaults (measured: ~108s) -- not a hang. ``sample_cartesian`` needs one
Bessel-kernel per distinct harmonic order (here, 64 for ``n_angular=127``),
each requiring ``n_radial`` zeros of that order's Bessel function; the cost
lands almost entirely in ``scipy.special.jn_zeros``, and unlike
``pypft.forward_pft``'s own per-order kernel cache, nothing here persists
between separate runs of this script, so every invocation pays it fresh.
"""

import argparse
import glob
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import EpsImagePlugin, Image

import pypft

#: Suffixes rasterized via Pillow's Ghostscript-backed EpsImagePlugin, rather
#: than loaded directly as a raster.
_VECTOR_SUFFIXES = {".eps", ".ps"}

#: Ghostscript binary names checked on PATH, in the same order
#: ``EpsImagePlugin.has_ghostscript`` itself checks them.
_GS_PATH_CANDIDATES = ("gswin32c", "gswin64c", "gs")

#: Where the official Windows Ghostscript installer places ``gswin64c.exe`` --
#: versioned, hence the glob, so a Ghostscript upgrade doesn't silently break
#: this fallback.
_WINDOWS_GS_GLOB = r"C:\Program Files\gs\gs*\bin\gswin64c.exe"


def _resolve_ghostscript() -> None:
    """Point Pillow's ``EpsImagePlugin`` at a Ghostscript binary.

    Duplicates (rather than calls) ``EpsImagePlugin.has_ghostscript``'s own
    ``PATH`` search: that function caches a failed result in its own
    module-level ``gs_binary`` the moment it is called once, and only ever
    recomputes when ``gs_binary is None`` -- so calling it first and then
    setting ``gs_windows_binary`` on a fallback path has no effect, since
    ``gs_binary`` itself is never updated. Setting both globals directly here
    sidesteps that caching
    entirely. The fallback below exists because a Ghostscript installed after
    this interpreter started (or by a silent installer that didn't refresh
    ``PATH``) is invisible to a plain ``PATH`` search either way.

    :raises RuntimeError: If Ghostscript is not found on ``PATH`` or at the
        default Windows install location.

    """
    for binary in _GS_PATH_CANDIDATES:
        if shutil.which(binary) is not None:
            EpsImagePlugin.gs_windows_binary = binary
            EpsImagePlugin.gs_binary = binary
            return
    if sys.platform.startswith("win"):
        candidates = sorted(glob.glob(_WINDOWS_GS_GLOB), reverse=True)
        if candidates:
            EpsImagePlugin.gs_windows_binary = candidates[0]
            EpsImagePlugin.gs_binary = candidates[0]
            return
    raise RuntimeError(
        "Ghostscript not found (checked PATH and the default Windows install "
        "directory) -- required to rasterize a .eps/.ps source."
    )


def _load_grayscale(path: Path, *, eps_scale: float) -> np.ndarray:
    """Load ``path`` as a 2-D ``uint8`` grayscale array, rasterizing vector sources.

    :param path: The source image path.
    :type path: Path
    :param eps_scale: Ghostscript's own rasterization scale, used only for a
        ``.eps``/``.ps`` source -- ``1.0`` renders at the file's own
        BoundingBox resolution.
    :type eps_scale: float
    :returns: The loaded ``(height, width)`` grayscale array.
    :rtype: np.ndarray
    :raises FileNotFoundError: If ``path`` does not exist or cannot be decoded.
    :raises RuntimeError: If ``path`` is a vector source and Ghostscript is
        not found.

    """
    if path.suffix.lower() in _VECTOR_SUFFIXES:
        _resolve_ghostscript()
        image = Image.open(path)
        # EpsImageFile.load's own `scale` keyword isn't in Image.load's base
        # type stub -- pyright sees only the base signature here.
        image.load(scale=eps_scale)  # type: ignore[call-arg]
        return np.asarray(image.convert("L"))
    array = cv2.imread(filename=str(path), flags=cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise FileNotFoundError(f"could not load image at {path}")
    return array


def _resize_square(image: np.ndarray, *, size: int) -> np.ndarray:
    """Resize ``image`` to a ``(size, size)`` square.

    ``INTER_AREA`` is used when shrinking (the common case, and the
    interpolation OpenCV itself recommends for downsampling); ``INTER_CUBIC``
    when enlarging instead.

    :param image: The source grayscale array.
    :type image: np.ndarray
    :param size: The output square's side length, in pixels.
    :type size: int
    :returns: The resized ``(size, size)`` array.
    :rtype: np.ndarray

    """
    interpolation = cv2.INTER_AREA if size <= max(image.shape) else cv2.INTER_CUBIC
    return cv2.resize(src=image, dsize=(size, size), interpolation=interpolation)


def _sample_polar(
    image: np.ndarray, *, n_radial: int, n_angular: int, r_fraction: float
) -> tuple[np.ndarray, pypft.PolarGrid]:
    """Sample ``image`` onto a ``PolarGrid``, printing ``check_adequacy``'s verdict.

    :param image: The square grayscale array to sample, ``(size, size)``.
    :type image: np.ndarray
    :param n_radial: The grid's radial sample count.
    :type n_radial: int
    :param n_angular: The grid's angular sample count.
    :type n_angular: int
    :param r_fraction: The grid's space limit ``R``, as a fraction of the
        image's own half-width -- ``0.95`` leaves a 5% margin so the sampled
        circle stays inside the square image.
    :type r_fraction: float
    :returns: The ``(n_radial, n_angular)`` polar-sampled array (rounded and
        clipped back to ``uint8``, since ``sample_cartesian`` itself
        interpolates in ``float64``), and the grid it was sampled on.
    :rtype: tuple[np.ndarray, pypft.PolarGrid]

    """
    size = image.shape[0]
    grid = pypft.PolarGrid(
        n_radial=n_radial, n_angular=n_angular, R=r_fraction * size / 2
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pypft.check_adequacy(grid=grid)
    if caught:
        for warning in caught:
            print(f"check_adequacy: {warning.category.__name__}: {warning.message}")
    else:
        print("check_adequacy: no warning raised (grid resolution is adequate)")
    sampled = pypft.sample_cartesian(image=image, grid=grid).T  # (n_radial, n_angular)
    return np.clip(np.round(sampled), 0, 255).astype(np.uint8), grid


def main() -> None:
    """Parse CLI arguments, rasterize/resize/sample the source, and save the result."""
    parser = argparse.ArgumentParser(
        description="Rasterize a source image and sample it onto a PolarGrid, "
        "for a test fixture."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source image path (.eps/.ps, or any raster format OpenCV can read).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the polar-sampled grayscale TIFF.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Square side length, in pixels, the source is resized to before "
        "polar sampling (default: 256, matching tests/samples/lena.tif).",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=1024,
        help="The PolarGrid's radial sample count (default: 1024).",
    )
    parser.add_argument(
        "--n-angular",
        type=int,
        default=127,
        help="The PolarGrid's angular sample count (default: 127).",
    )
    parser.add_argument(
        "--r-fraction",
        type=float,
        default=0.95,
        help="The PolarGrid's space limit R, as a fraction of the resized "
        "image's own half-width (default: 0.95, a 5%% margin).",
    )
    parser.add_argument(
        "--eps-scale",
        type=float,
        default=1.0,
        help="Ghostscript rasterization scale for a .eps/.ps source, applied "
        "before resizing to --size (default: 1.0, the source's own "
        "BoundingBox resolution).",
    )
    args = parser.parse_args()

    image = _load_grayscale(args.input, eps_scale=args.eps_scale)
    print(f"Loaded {args.input} ({image.shape[1]}x{image.shape[0]}, {image.dtype})")

    resized = _resize_square(image, size=args.size)
    polar, grid = _sample_polar(
        resized,
        n_radial=args.n_radial,
        n_angular=args.n_angular,
        r_fraction=args.r_fraction,
    )
    print(f"Grid: n_radial={grid.n_radial}, n_angular={grid.n_angular}, R={grid.R:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(polar, mode="L").save(args.output, compression="none")
    print(
        f"Wrote {args.output} ({polar.shape[1]}x{polar.shape[0]}, "
        "uint8 grayscale TIFF, (n_radial, n_angular) layout)"
    )


if __name__ == "__main__":
    main()
