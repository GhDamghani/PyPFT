import numpy as np
from pypft.tranform import inverse
from pypft.utils.polar import diagonal_to_radial
from pypft.utils.visualization import visualize_reconstructed, visualize_kspace, show
import sys
from pathlib import Path

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        """
        Test data from:
            El‐Rewaidy, H., Fahmy, A. S., Pashakhanloo, F., Cai, X., Kucukseymen, S., Csecs, I., ... & Nezafat, R. (2021).
            Multi‐domain convolutional neural network (MD‐CNN) for radial reconstruction of dynamic cardiac MRI.
            Magnetic Resonance in Medicine, 85(3), 1195-1208."""
        
        test_data = np.load(Path(__file__).parent/'test_data.npy')
        test_data = diagonal_to_radial(test_data)
        result = inverse(test_data)

        visualize_kspace(test_data, 'K-Space')
        visualize_reconstructed(result, 'Reconstructed')
        show()
    else:
        print("Usage: python -m pypft test")

if __name__ == "__main__":
    main()