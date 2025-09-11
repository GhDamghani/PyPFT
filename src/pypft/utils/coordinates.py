import cv2 as cv

def polar_to_cart(img_polar, img_square_size=None):
    """Convert a polar image to a Cartesian image using OpenCV's warpPolar function.
    img_polar: 2D numpy array, polar image. shape = (samples, spokes)
    img_square_size: int, the size of the output square image. If None, it will be set to samples.
    return:
    img_cart: 2D numpy array, Cartesian image. shape = (img_square_size, img_square_size)
    """
    if img_square_size is None:
        img_square_size = img_polar.shape[0]

    img_cart = cv.warpPolar(
        img_polar.T,
        (img_square_size, img_square_size),
        (img_square_size // 2, img_square_size // 2),
        img_square_size / 2,
        cv.WARP_INVERSE_MAP,
    )

    assert img_cart.shape == (img_square_size, img_square_size)
    return img_cart