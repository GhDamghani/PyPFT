"""Tests for ``pypft.samples``, the test-sample downloader.

Nothing here touches the network: every download is of a local file through a
``file://`` URL, and the MD-CNN excerpt is cut from a small synthetic array in the
dataset's layout.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

import pypft.samples as samples
from pypft.samples import (
    ChecksumError,
    RemoteFile,
    Sample,
    _md_cnn,
    download_file,
    fetch_sample,
)


def _data() -> np.ndarray:
    """A ``(2, 3, 12, 4, 5, 2)`` array in the dataset's layout, coil 3 the strongest.

    :returns: The array.
    :rtype: np.ndarray

    """
    data = np.random.default_rng(seed=0).standard_normal(size=(2, 3, 12, 4, 5, 2))
    data[..., 3, :] *= 10.0
    data[..., 1, :] *= 5.0
    return data


def _remote(path: Path, *, filename: str = "source.bin") -> RemoteFile:
    """A ``RemoteFile`` for a local file, with that file's own checksum."""
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    return RemoteFile(url=path.as_uri(), filename=filename, md5=md5)


# ========================================================================================
# The MD-CNN excerpt
# ========================================================================================


def test_the_excerpt_keeps_the_layout_and_the_center() -> None:
    data = _data()
    excerpt = _md_cnn.make_excerpt(data=data, readout=8, coils=2)
    assert excerpt.shape == (1, 1, 8, 4, 2, 2)
    assert excerpt.dtype == np.float32
    # The full readout's center, index 6, is the excerpt's center, index 4; the two
    # strongest coils, 1 and 3, in coil order.
    np.testing.assert_array_equal(
        excerpt[0, 0, 4], data[0, 0, 6][:, [1, 3]].astype(np.float32)
    )


@pytest.mark.parametrize(
    ("readout", "coils"), [(7, 2), (14, 2), (8, 6), (0, 2), (8, 0)]
)
def test_invalid_excerpt_sizes_are_rejected(readout: int, coils: int) -> None:
    with pytest.raises(ValueError):
        _md_cnn.make_excerpt(data=_data(), readout=readout, coils=coils)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"data": [[1.0]]}, TypeError),
        ({"data": np.ones((2, 3, 12, 4, 5))}, ValueError),
        ({"readout": 8.0}, TypeError),
        ({"coils": "2"}, TypeError),
    ],
)
def test_invalid_excerpt_arguments_are_rejected(kwargs: dict, error: type) -> None:
    arguments = {"data": _data(), "readout": 8, "coils": 2, **kwargs}
    with pytest.raises(error):
        _md_cnn.make_excerpt(**arguments)


# ========================================================================================
# Downloads
# ========================================================================================


def test_download_file_copies_and_checks_the_file(tmp_path: Path) -> None:
    source = tmp_path / "published.bin"
    source.write_bytes(b"k-space" * 1000)
    target = download_file(remote=_remote(source), directory=tmp_path / "cache")
    assert target == tmp_path / "cache" / "source.bin"
    assert target.read_bytes() == source.read_bytes()


def test_download_file_keeps_a_file_with_the_right_checksum(tmp_path: Path) -> None:
    source = tmp_path / "published.bin"
    source.write_bytes(b"k-space")
    remote = _remote(source)
    download_file(remote=remote, directory=tmp_path / "cache")
    # Gone from its source, the file is still found in the directory.
    source.unlink()
    target = download_file(remote=remote, directory=tmp_path / "cache")
    assert target.read_bytes() == b"k-space"


def test_a_wrong_checksum_leaves_no_file(tmp_path: Path) -> None:
    source = tmp_path / "published.bin"
    source.write_bytes(b"k-space")
    remote = RemoteFile(url=source.as_uri(), filename="source.bin", md5="0" * 32)
    with pytest.raises(ChecksumError):
        download_file(remote=remote, directory=tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_download_file_rejects_a_wrong_remote(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        download_file(remote="https://example.org", directory=tmp_path)  # type: ignore


# ========================================================================================
# Fetching samples
# ========================================================================================


@pytest.fixture
def synthetic_md_cnn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Point ``Sample.MD_CNN_EXCERPT``'s recipe at a small local dataset file.

    :returns: The file's ``data`` array.
    :rtype: np.ndarray

    """
    data = np.random.default_rng(seed=1).standard_normal(size=(1, 2, 832, 3, 6, 2))
    source = tmp_path / "published.mat"
    savemat(file_name=source, mdict={_md_cnn.VARIABLE: data})
    recipe = samples._RECIPES[Sample.MD_CNN_EXCERPT]
    patched = samples._Recipe(
        source=_remote(source, filename=_md_cnn.SOURCE_FILENAME), make=recipe.make
    )
    monkeypatch.setitem(samples._RECIPES, Sample.MD_CNN_EXCERPT, patched)
    return data


def test_fetch_sample_writes_the_md_cnn_excerpt(
    tmp_path: Path, synthetic_md_cnn: np.ndarray
) -> None:
    output = fetch_sample(sample=Sample.MD_CNN_EXCERPT, directory=tmp_path / "out")
    assert output == tmp_path / "out" / Sample.MD_CNN_EXCERPT.value
    np.testing.assert_array_equal(
        loadmat(file_name=output)[_md_cnn.VARIABLE],
        _md_cnn.make_excerpt(
            data=synthetic_md_cnn,
            readout=_md_cnn.EXCERPT_READOUT,
            coils=_md_cnn.EXCERPT_COILS,
        ),
    )
    # Only the sample is left: no partial file, no source.
    assert list(output.parent.iterdir()) == [output]


def test_fetch_sample_keeps_the_source_in_the_cache_directory(
    tmp_path: Path, synthetic_md_cnn: np.ndarray
) -> None:
    fetch_sample(
        sample=Sample.MD_CNN_EXCERPT,
        directory=tmp_path / "out",
        cache_directory=tmp_path / "cache",
    )
    assert (tmp_path / "cache" / _md_cnn.SOURCE_FILENAME).is_file()


def test_fetch_sample_keeps_an_existing_sample_unless_overwritten(
    tmp_path: Path, synthetic_md_cnn: np.ndarray
) -> None:
    output = tmp_path / Sample.MD_CNN_EXCERPT.value
    output.write_bytes(b"placeholder")
    assert fetch_sample(sample=Sample.MD_CNN_EXCERPT, directory=tmp_path) == output
    assert output.read_bytes() == b"placeholder"
    fetch_sample(sample=Sample.MD_CNN_EXCERPT, directory=tmp_path, overwrite=True)
    assert loadmat(file_name=output)[_md_cnn.VARIABLE].shape[2] == 416


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"sample": "MD_CNN_EXCERPT"}, ValueError),
        ({"directory": "out"}, TypeError),
        ({"cache_directory": "cache"}, TypeError),
        ({"overwrite": 1}, TypeError),
    ],
)
def test_fetch_sample_rejects_invalid_arguments(
    tmp_path: Path, kwargs: dict, error: type
) -> None:
    arguments = {"sample": Sample.MD_CNN_EXCERPT, "directory": tmp_path, **kwargs}
    with pytest.raises(error):
        fetch_sample(**arguments)


def test_main_fetches_every_sample_by_default(
    tmp_path: Path, synthetic_md_cnn: np.ndarray, capsys: pytest.CaptureFixture
) -> None:
    assert samples.main(argv=["--directory", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / Sample.MD_CNN_EXCERPT.value).is_file()
    assert "MD_CNN_EXCERPT" in capsys.readouterr().out


def test_every_sample_has_a_recipe() -> None:
    assert set(samples._RECIPES) == set(Sample)
