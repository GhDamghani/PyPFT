import matplotlib.pyplot as plt
import numpy as np

def plot_bessel_mat(x, title=None):
    """Plot complex image.
    input:
    x: 2D numpy array"""

    assert x.ndim == 2
    assert x.dtype.kind in 'c'

    assert type(title) == str or title is None
    
    x = np.abs(x)
    x = x / np.max(x)
    x = x ** 0.3

    plt.figure()
    plt.imshow(x, cmap='gray')
    plt.title(title)
def plot_bessel_mat(im, title=None):
    """Plot bessel matrix.
    input:
    im: 2D numpy array"""

    assert im.ndim == 2
    assert im.dtype.kind in 'uif'

    assert type(title) == str or title is None
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title(title)


def plot_image(im, title=None):

    """Plot a given image.
    input:
    im: 2D numpy array"""

    assert im.ndim in (2, 3)
    assert im.dtype.kind in 'uif'

    assert type(title) == str or title is None
    
    plt.figure()
    plt.imshow(im, cmap='gray' if im.ndim == 2 else None)
    plt.title(title)

def circle_frame(im):
    """Putting a circle frame on a square image.
    input:
    x: 2D numpy array, square image."""
    assert im.ndim == 2
    assert im.shape[0] == im.shape[1]
    assert im.dtype.kind in 'uif'

    x = np.arange(im.shape[0])
    y = np.arange(im.shape[1])
    x, y = np.meshgrid(x, y)
    mask = (x - im.shape[0] // 2) ** 2 + (y - im.shape[1] // 2) ** 2 < (im.shape[0] // 2) ** 2
    im = im / np.max(im)
    im = np.dstack([im] * 3 + [mask])
    return im

def visualize_reconstructed(f, title=None):
    """Given a 2D numpy array of reconstructed image in polar coordinates, plot it in Cartesian coordinates.
    input:
    f: 2D numpy array, in the shape of (samples, spokes)."""

    assert f.ndim == 2
    assert f.shape[1] % 2 == 0

    from .coordinates import polar_to_cart

    f = np.abs(f)
    f = polar_to_cart(f)
    f = circle_frame(f)
    plot_image(f, title)

def show():
    plt.show()