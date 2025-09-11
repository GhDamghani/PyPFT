from unittest import TestCase
import numpy as np
from src.pypft.tranform import inverse
from src.pypft.utils.polar import diagonal_to_radial
from src.pypft.utils.visualization import visualize_reconstructed, show
from src.pypft.bessel_mat_handler import filename_bessel_mat, delete_bessel_mat_cache_npy

class TestTransformations(TestCase):
    def setUp(self):
        self.test_data = np.load('tests/test_files/001_00.npy') # Sample data
        self.test_data_recon = np.load('tests/test_files/001_00_recon.npy') # Sample reconstructed data
        self.bessel_mat = read_bessel_mat_bin('tests/test_files/bessel_mat_208_392.bin') # Computed bessel matrix, implemented in the Future
        self.DHT_impls = ['naive', 'parallel', 'arrayprogramming']
        self.bessel_mat_impls = ['naive', 'symmetric']
    
    def test_inverse_sample(self):
        test_data = diagonal_to_radial(self.test_data)

        for DHT_impl in self.DHT_impls:
            result = inverse(test_data, bessel_mat=self.bessel_mat, DHT_impl=DHT_impl)

            # compare_results_vis(result, self.test_data_recon)

            self.assertEqual(result.shape, self.test_data_recon.shape)
            self.assertAlmostEqual(np.linalg.norm(result - self.test_data_recon), 0, places=5)
    
    def test_inverse_random(self):
        sample, spoke = np.random.randint(1, 128), np.random.randint(1, 64) * 2
        test_data = np.random.rand(sample, spoke)

        cart_r = int(np.round(sample * np.sqrt(2)))

        # Making sure the bessel matrix is not cached
        filename = filename_bessel_mat(sample, spoke)
        delete_bessel_mat_cache_npy(filename)

        for bessel_mat_impl in self.bessel_mat_impls:
           result = inverse(test_data, bessel_mat=bessel_mat_impl)
           self.assertEqual(result.shape, (cart_r, spoke))
        
        # Deleting after using
        delete_bessel_mat_cache_npy(filename)



def read_bessel_mat_bin(filename):
    
    with open(filename, "rb") as f:
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        bessel_mat = np.fromfile(f, dtype=np.float64).reshape((rows, cols))
    return bessel_mat

def compare_results_vis(recon, org):
    """Compare reconstructed and original data, visually."""
    visualize_reconstructed(recon, title='Reconstructed')
    visualize_reconstructed(org, title='Original')
    show()