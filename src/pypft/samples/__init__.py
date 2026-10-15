"""Download PyPFT's test samples from their public sources.

Large measured samples are not part of the repository or the package: each one is
downloaded from where its owners publish it, checked against the published checksum,
and, when only part of the published file is needed, cut down to that part locally.
``fetch_sample`` does all of it for one ``Sample``; ``python -m pypft.samples`` does it
from the command line.

A sample is described by a ``RemoteFile`` (where its source is published, under which
name, with which MD5 checksum) and a function writing the sample from that source,
registered together in ``_RECIPES``. Adding a sample means adding a ``Sample`` member
and its recipe; nothing else changes.

The samples so far:

- ``Sample.MD_CNN_EXCERPT``: one slice, one cardiac phase, the central 416 readout
  samples and the four strongest coils of subject P01 of the MD-CNN radial cine
  dataset (Harvard Dataverse, doi:10.7910/DVN/CI3WB6, CC0 1.0), in the dataset's own
  layout (``pypft.acquisition.RadialLineLayout.SLICE_PHASE_READOUT_LINE_COIL_COMPLEX``).
  The source is about 230 MB; the sample, 1.9 MB.
"""

import argparse
import hashlib
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.request import Request, urlopen

from pypft.samples import _md_cnn
from pypft.utils.validators import EnumValidator, PathValidator, StrValidator

# ========================================================================================
# Constants
# ========================================================================================

#: Bytes read from the network, and from a file being hashed, at a time.
_CHUNK_SIZE = 1 << 20

#: Seconds without a response before a download fails.
DOWNLOAD_TIMEOUT = 60.0

#: The ``User-Agent`` header of every download: some hosts, Harvard Dataverse among
#: them, refuse ``urllib``'s default one.
USER_AGENT = "pypft"

#: The suffix of a download that has not been checked yet.
_PARTIAL_SUFFIX = ".part"

# ========================================================================================
# Errors
# ========================================================================================


class ChecksumError(ValueError):
    """A downloaded file's MD5 checksum is not the one its source publishes."""


# ========================================================================================
# The samples
# ========================================================================================


class Sample(Enum):
    """A test sample, by the file name it is written under."""

    MD_CNN_EXCERPT = "P01_slc1_phs1_lins196_cols416_cha4.mat"


@dataclass(frozen=True)
class RemoteFile:
    """A file published for download, with the checksum its publisher gives.

    :param url: Where the file is downloaded from.
    :type url: str
    :param filename: The name it is stored under.
    :type filename: str
    :param md5: Its MD5 checksum, in hexadecimal.
    :type md5: str

    """

    url: str
    filename: str
    md5: str


@dataclass(frozen=True)
class _Recipe:
    """How one sample is made: its source, and the function writing it from there."""

    source: RemoteFile
    make: Callable[[Path, Path], None]


#: Every sample's recipe; ``make(source, output)`` writes the sample to ``output``.
_RECIPES: dict[Sample, _Recipe] = {
    Sample.MD_CNN_EXCERPT: _Recipe(
        source=RemoteFile(
            url=_md_cnn.SOURCE_URL,
            filename=_md_cnn.SOURCE_FILENAME,
            md5=_md_cnn.SOURCE_MD5,
        ),
        make=_md_cnn.write_excerpt,
    ),
}

# ========================================================================================
# Downloads
# ========================================================================================


def _md5(path: Path) -> str:
    """Return a file's MD5 checksum in hexadecimal."""
    digest = hashlib.md5()
    with path.open(mode="rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(remote: RemoteFile, *, directory: Path) -> Path:
    """Download a file into a directory and check its checksum.

    A file of the same name already in ``directory`` with the right checksum is kept
    rather than downloaded again. The download goes to a ``.part`` file first, renamed
    only once its checksum matches, so an interrupted or corrupt download never leaves
    a file under the final name.

    :param remote: The file to download.
    :type remote: RemoteFile
    :param directory: Where it is stored; created when missing.
    :type directory: Path
    :returns: The downloaded file.
    :rtype: Path
    :raises TypeError: If an argument has the wrong type.
    :raises OSError: If ``directory`` cannot be written, or the download fails.
    :raises ChecksumError: If the downloaded file's checksum is not ``remote.md5``.

    """
    if not isinstance(remote, RemoteFile):
        raise TypeError(f"remote must be a RemoteFile, got {type(remote).__name__}")
    StrValidator.type_is_str(value=remote.url)
    StrValidator.type_is_str(value=remote.filename)
    StrValidator.type_is_str(value=remote.md5)
    PathValidator.type_is_Path(value=directory)
    PathValidator.value_is_writable(value=directory)

    target = directory / remote.filename
    if target.is_file() and _md5(path=target) == remote.md5:
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + _PARTIAL_SUFFIX)
    digest = hashlib.md5()
    try:
        with (
            urlopen(
                Request(url=remote.url, headers={"User-Agent": USER_AGENT}),
                timeout=DOWNLOAD_TIMEOUT,
            ) as response,
            partial.open(mode="wb") as stream,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != remote.md5:
            raise ChecksumError(
                f"{remote.url} has MD5 {digest.hexdigest()}, expected {remote.md5}"
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def fetch_sample(
    sample: Sample,
    *,
    directory: Path,
    cache_directory: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a test sample's source and write the sample into a directory.

    :param sample: The sample.
    :type sample: Sample
    :param directory: Where the sample is written, as ``sample.value``; created when
        missing.
    :type directory: Path
    :param cache_directory: Where the downloaded source is kept, so the next call
        does not download it again; a temporary directory, removed afterwards, when
        omitted.
    :type cache_directory: Path | None
    :param overwrite: Write the sample even when ``directory`` already holds it;
        otherwise an existing sample is returned as it is, with nothing downloaded.
    :type overwrite: bool
    :returns: The sample's file.
    :rtype: Path
    :raises TypeError: If an argument has the wrong type.
    :raises ValueError: If ``sample`` is not a ``Sample`` member.
    :raises OSError: If a directory cannot be written, or the download fails.
    :raises ChecksumError: If the downloaded source's checksum is not the published
        one.

    """
    EnumValidator.value_is_enum_member(value=sample, enum_class=Sample)
    PathValidator.type_is_Path(value=directory)
    PathValidator.value_is_writable(value=directory)
    if cache_directory is not None:
        PathValidator.type_is_Path(value=cache_directory)
        PathValidator.value_is_writable(value=cache_directory)
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be a bool, got {type(overwrite).__name__}")

    output = directory / sample.value
    if output.is_file() and not overwrite:
        return output
    recipe = _RECIPES[sample]
    directory.mkdir(parents=True, exist_ok=True)
    # The sample is written under a temporary name and renamed when complete, like a
    # download; the name keeps the sample's suffix, which some writers append.
    partial = output.with_name(f"{output.stem}{_PARTIAL_SUFFIX}{output.suffix}")
    try:
        if cache_directory is None:
            with tempfile.TemporaryDirectory() as temporary:
                source = download_file(remote=recipe.source, directory=Path(temporary))
                recipe.make(source, partial)
        else:
            source = download_file(remote=recipe.source, directory=cache_directory)
            recipe.make(source, partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output


# ========================================================================================
# Command line
# ========================================================================================


def main(argv: list[str] | None = None) -> int:
    """Fetch test samples from the command line.

    :param argv: The command-line arguments; ``sys.argv[1:]`` when omitted.
    :type argv: list[str] | None
    :returns: The exit code, ``0``.
    :rtype: int

    """
    names = [sample.name for sample in Sample]
    parser = argparse.ArgumentParser(
        prog="python -m pypft.samples",
        description="Download PyPFT's test samples from their public sources.",
    )
    parser.add_argument(
        "samples",
        nargs="*",
        choices=names,
        metavar="SAMPLE",
        help=f"The samples to fetch: {', '.join(names)} (default: all).",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        required=True,
        help="Where the samples are written.",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=None,
        help="Where downloaded sources are kept (default: not kept).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write samples that are already present again.",
    )
    arguments = parser.parse_args(args=argv)
    # No sample named means every sample; one named twice is fetched once.
    for name in dict.fromkeys(arguments.samples or names):
        path = fetch_sample(
            sample=Sample[name],
            directory=arguments.directory,
            cache_directory=arguments.cache_directory,
            overwrite=arguments.overwrite,
        )
        print(f"{name}: {path}")
    return 0
