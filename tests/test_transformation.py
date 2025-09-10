from unittest import TestCase
import numpy as np
from src.PyPFT.tranform import inverse
from src.PyPFT.utils.polar import diagonal_to_radial

def read_bessel_mat_bin(filename):
    
    with open(filename, "rb") as f:
        rows = np.fromfile(f, dtype=np.int32, count=1)[0]
        cols = np.fromfile(f, dtype=np.int32, count=1)[0]
        bessel_mat = np.fromfile(f, dtype=np.float64).reshape((rows, cols))
    return bessel_mat

class TestTransformations(TestCase):
    def setUp(self):
        self.test_data = np.load('tests/test_files/001_00.npy') # Sample data
        self.test_data_recon = np.load('tests/test_files/001_00_recon.npy') # Sample reconstructed data
        self.bessel_mat = read_bessel_mat_bin('tests/test_files/bessel_mat_208_392.bin') # Computed bessel matrix, implemented in the Future
    
    def test_inverse(self):
        test_data = diagonal_to_radial(self.test_data)

        DHT_impls = ['naive', 'parallel']

        for DHT_impl in DHT_impls:
            result = inverse(test_data, self.bessel_mat, DHT_impl=DHT_impl)

            self.assertEqual(result.shape, self.test_data_recon.shape)
            self.assertAlmostEqual(np.linalg.norm(result - self.test_data_recon), 0, places=5)