from pathlib import Path
from platformdirs import user_cache_dir
import numpy as np
import src.pypft.bessel_mat_impls as bessel_mat_impls

cache_dir = Path(user_cache_dir("PyPFT"))

def open_cache_folder():
    import sys
    import subprocess
    cache_dir.mkdir(parents=True, exist_ok=True)  # make sure it exists

    if sys.platform.startswith("win"):
        subprocess.run(["explorer", str(cache_dir)])
    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", str(cache_dir)])
    else:  # assume Linux / Unix
        subprocess.run(["xdg-open", str(cache_dir)])

def filename_bessel_mat(samples, spokes):
    """Generate the filename for a Bessel matrix file based on the number of samples and spokes.
    
    Parameters:
    ----------
    samples : int
        The number of samples.
    spokes : int
        The number of spokes.
    
    Returns:
    -------
    filename : str
        The filename for the Bessel matrix file.
    """
    assert type(samples) == int
    assert type(spokes) == int
    assert spokes % 2 == 0
    assert samples > 0
    assert spokes > 0

    filename = f"bessel_mat_{samples}_{spokes}"
    return filename


def delete_bessel_mat_cache_npy(filename):
    """Delete a Bessel matrix from the cache directory as a numpy .npy file.
    
    Parameters:
    ----------
    filename : str
        The name of the Bessel matrix file to delete.
    """
    bessel_file = cache_dir / filename
    if bessel_file.exists():
        bessel_file.unlink()

def load_bessel_mat(samples, spokes, bessel_mat_impl=None):
    """Based on the number of samples and spokes, load the Bessel matrix from cache or calculate the Bessel matrix and save it to cache folder.
    input:
        samples: number of samples
        spokes: number of spokes. even number
        bessel_mat_impl: implementation of the bessel matrix. If None, use the default implementation (sym_vect_par).
    output:
        bessel_mat: 2D numpy array, in the shape of (samples * cart_r, spokes). cart_r = int(np.round(samples * np.sqrt(2)))
    """
    filename = f"bessel_mat_{samples}_{spokes}"
    bessel_mat = load_bessel_mat_cache_npy(filename)
    if bessel_mat is None:
        bessel_mat_impl = 'sym_vect_par' if bessel_mat_impl is None else bessel_mat_impl
        bessel_mat = getattr(bessel_mat_impls, bessel_mat_impl).bessel_mat(samples, spokes)
        save_bessel_mat_cache_npy(bessel_mat, filename)
    return bessel_mat

def read_bessel_mat_bin(filename):
    with open(filename, "rb") as f:
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        bessel_mat = np.fromfile(f, dtype=np.float64).reshape((rows, cols))
    return bessel_mat

def save_bessel_mat_cache_npy(bessel_mat, filename):
    
    """
    Save a Bessel matrix to the cache directory as a numpy .npy file.
    
    Parameters
    ----------
    bessel_mat : 2D numpy array
        The Bessel matrix to save.
    filename : str
        The filename to use for the saved file, without extension.
    """
    assert bessel_mat.ndim == 2
    assert bessel_mat.dtype.kind == 'f'
    assert bessel_mat.shape[1] % 2 == 0
    assert type(filename) == str
    assert not filename.endswith('.npy')

    cache_dir.mkdir(parents=True, exist_ok=True)

    bessel_file = cache_dir / filename
    np.save(bessel_file, bessel_mat)

def load_bessel_mat_cache_npy(filename):
    
    """
    Load a Bessel matrix from the cache directory as a numpy .npy file.
    
    Parameters
    ----------
    filename : str
        The filename to use for the saved file, with extension.
    
    Returns
    -------
    bessel_mat : 2D numpy array or None
        The loaded Bessel matrix, or None if it does not exist in the cache.
    """
    assert type(filename) == str
    assert not filename.endswith('.npy')

    bessel_file = cache_dir / (filename + '.npy')
    if not bessel_file.exists():
        return None
    return np.load(bessel_file)