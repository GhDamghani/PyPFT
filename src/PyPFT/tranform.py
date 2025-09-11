from numpy.fft import fft, ifft, fftshift, ifftshift
import numpy as np
from . import DHT_impls
from . import bessel_mat_handler

def inverse(F_rho_phi, bessel_mat=None, DHT_impl=None):
    """Inverse Polar Fourier Transform
    F_rho_phi: 2D numpy array in shape of (samples, spokes). spokes are radial, NOT diagonal, therefore spokes is an even number
    bessel_mat:
        None: load or calculate bessel matrix on the fly, based on the default implementation
        str: load or calculate bessel matrix on the fly, based on the given implementation
        2D numpy array: use the given bessel matrix
    
    DHT_impl:
        None: use the default implementation (parallel)
        str: use the implementation with the given name

    return:
    f_r_theta: 2D numpy array in the shape of (samples, spokes) spokes are radial, NOT diagonal, therefore spokes is an even number

    Steps:
    \[F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)\]
    """
    assert F_rho_phi.ndim == 2
    assert F_rho_phi.shape[1] % 2 == 0
    assert bessel_mat is None or type(bessel_mat) == str or bessel_mat.ndim == 2
    assert DHT_impl is None or DHT_impl in ['naive', 'parallel', 'arrayprogramming']

    samples, spokes = F_rho_phi.shape

    if bessel_mat is None:
        bessel_mat = bessel_mat_handler.bessel_mat(samples, spokes)
    elif type(bessel_mat) == str:
        bessel_mat_impl = bessel_mat
        bessel_mat = bessel_mat_handler.load_bessel_mat(samples, spokes, bessel_mat_impl)
    if bessel_mat.ndim == 2:
        cart_r = int(np.round(samples * np.sqrt(2)))
        assert bessel_mat.shape == (cart_r * samples, spokes)
    
    # 1. FFT along the angular direction
    F_rho_n = fftshift(fft(F_rho_phi, axis=1), 1) * (2 * np.pi * samples)

    # 2. Hankel transform along the radial direction
    match DHT_impl:
        case None:
            DHT = DHT_impls.parallel
        case 'naive':
            DHT = DHT_impls.naive
        case 'parallel':
            DHT = DHT_impls.parallel
        case 'arrayprogramming':
            DHT = DHT_impls.arrayprogramming
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