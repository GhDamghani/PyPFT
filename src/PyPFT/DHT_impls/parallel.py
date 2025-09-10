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

    r_indx = int(np.round(samples * np.sqrt(2)))
    F = np.zeros((r_indx, spokes), "complex128")
    def calc_F(ord_n, w, k, f_rho, bessel_mat):
        return np.sum(f_rho[:, ord_n] * bessel_mat[w: k, ord_n])

    F = np.array(Parallel(n_jobs=-1)(delayed(calc_F)(ord_n, samples * j, samples * (j + 1), f_rho, bessel_mat) for j in range(r_indx) for ord_n in range(spokes))).reshape((r_indx, spokes))
    
    return F