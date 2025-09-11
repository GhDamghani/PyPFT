import numpy as np
from joblib import Parallel, delayed

def transform(f, bessel_mat):
    """Parallel DHT implementation.
    Requires joblib
    f: 2D numpy array, in the shape of (samples, spokes). spokes is an even number.
    
    return:
    F: 2D numpy array
    """
    assert f.ndim == 2
    assert f.shape[1] % 2 == 0
    samples, spokes = f.shape 

    rho = (1 / 2 / samples) + np.arange(0, 1, 1 / samples)
    f_rho = rho[:, np.newaxis] * f  # normalized rho

    cart_r = int(np.round(samples * np.sqrt(2)))
    F = np.zeros((cart_r, spokes), "complex128")
    def sub_transform(j):
        return np.sum(f_rho * bessel_mat[samples * j: samples * (j + 1)], axis=0)

    F = np.array(Parallel(n_jobs=-1)(delayed(sub_transform)(j) for j in np.arange(cart_r)))
    
    return F