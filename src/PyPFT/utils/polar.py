import numpy as np

def diagonal_to_radial(x):
    """Convert a 2D array with diagonal spokes to radial spokes.
    x: 2D numpy array, in the shape of (diag_samples, diag_spokes). diag_samples is even.
    return:
    y: 2D numpy array, in the shape of (samples, spokes), samples = diag_samples/2, spokes = diag_spokes*2
    """

    assert x.ndim == 2
    assert x.shape[0] % 2 == 0
    diag_samples, diag_spokes = x.shape
    samples = diag_samples // 2
    spokes = diag_spokes * 2
    x1 = x[:samples, :diag_spokes][::-1]
    x2 = x[samples:, :diag_spokes]
    y = np.concatenate((x1, x2), axis=1)
    assert y.shape == (samples, spokes)
    return y