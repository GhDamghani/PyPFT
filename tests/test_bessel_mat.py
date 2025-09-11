from unittest import TestCase
import numpy as np
import src.pypft.bessel_mat_impls as bessel_mat_impls
from src.pypft.bessel_mat_handler import filename_bessel_mat, delete_bessel_mat_cache_npy, load_bessel_mat
from src.pypft.utils.visualization import plot_bessel_mat, show

def read_bessel_mat_bin(filename):
    
    with open(filename, "rb") as f:
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        bessel_mat = np.fromfile(f, dtype=np.float64).reshape((rows, cols))
    return bessel_mat

class TestBesselMat(TestCase):
    def setUp(self):
        self.sample_bessel_mat = read_bessel_mat_bin('tests/test_files/bessel_mat_30_50.bin') # Computed bessel matrix
    
    def test_bessel_mat_sample(self):
        """
        Test that the bessel matrix implementation gives the same result as the precalculated
        bessel matrix.
        """
        samples, spokes = 30, 50

        for bessel_mat_impl in ['naive', 'symmetric', 'sym_vect', 'sym_vect_par']:
            bessel_mat = getattr(bessel_mat_impls, bessel_mat_impl).bessel_mat

            result = bessel_mat(samples, spokes)

            self.assertEqual(result.shape, self.sample_bessel_mat.shape)
            self.assertAlmostEqual(np.linalg.norm(result - self.sample_bessel_mat)/result.size, 0, places=4)
    
    def test_bessel_mat_random(self):
        for i in range(5):
            samples, spokes = np.random.randint(1, 128), np.random.randint(1, 64) * 2
            cart_r = int(np.round(samples * np.sqrt(2)))

            # Making sure the bessel matrix is not cached
            filename = filename_bessel_mat(samples, spokes)
            delete_bessel_mat_cache_npy(filename)

            bessel_mat = load_bessel_mat(samples, spokes)
            self.assertEqual(bessel_mat.shape, (cart_r * samples, spokes))

            # Deleting after using
            delete_bessel_mat_cache_npy(filename)

def compare_results_vis(recon, org):
    """Compare reconstructed and original data, visually."""
    plot_bessel_mat(recon, title='Reconstructed')
    plot_bessel_mat(org, title='Original')
    show()