import numpy as np
def transform(f, bessel_mat):
    """Naive DHT implementation.
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
    for ord_n in np.arange(spokes):
        for j in np.arange(r_indx):
            w = samples * j
            k = samples * (j + 1)
            F[j, ord_n] = np.sum(f_rho[:, ord_n] * bessel_mat[w: k, ord_n])
    
    return F