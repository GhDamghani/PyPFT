from numpy.fft import fft, ifft, fftshift, ifftshift
import numpy as np
from . import DHT_impls

def inverse(F_rho_phi, bessel_mat, DHT_impl='naive'):
    """Inverse Polar Fourier Transform
    F_rho_phi: 2D numpy array

    return:
    f_r_theta: 2D numpy array in the shape of (samples, spokes) spokes are radial, NOT diagonal, therefore spokes is an even number

    Steps:
    \[F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)\]
    """
    assert F_rho_phi.ndim == 2
    assert F_rho_phi.shape[1] % 2 == 0

    samples, spokes = F_rho_phi.shape
    
    # 1. FFT along the angular direction
    F_rho_n = fftshift(fft(F_rho_phi, axis=1), 1) * (2 * np.pi * samples)

    # 2. Hankel transform along the radial direction
    match DHT_impl:
        case 'naive':
            DHT = DHT_impls.naive
        case 'parallel':
            DHT = DHT_impls.parallel
    f_r_n = DHT.transform(F_rho_n, bessel_mat) # Placeholder for Hankel transform implementation

    # 3. IFFT along the angular direction
    f_r_theta = ifftshift(ifft(f_r_n, axis=1), 1) * (np.pi * samples)

    # assert f_r_theta.shape == (samples, spokes)
    return f_r_theta

def forward(f_r_theta):
    """Forward Polar Fourier Transform
    f_r_theta: 2D numpy array in the shape of (samples, spokes) spokes are radial, NOT diagonal, therefore spokes is an even number
    return:
    F_rho_phi: 2D numpy array
    Steps are the reverse of inverse()"""

    assert f_r_theta.ndim == 2
    assert f_r_theta.shape[1] % 2 == 0

    samples, spokes = f_r_theta.shape

    # 1. FFT along the angular direction
    f_r_n = fftshift(fft(f_r_theta, axis=1), 1)


    # 2. Hankel transform along the radial direction (Inverse of Hankel transformation is itself)
    F_rho_n = f_r_n # Placeholder for Inverse Hankel transform implementation

    # 3. IFFT along the angular direction
    F_rho_phi = ifftshift(ifft(F_rho_n, axis=1), 1)

    assert F_rho_phi.shape == (samples, spokes)
    return F_rho_phi