# PyPFT
Polar Fourier Transform for Reconstruction of Polar MR images in Python using Numpy

Assuming you have raw kspace data in polar coordinates $F\left( {\rho ,\varphi } \right)$, if you want to reconstruct the image in the spatial domain with polar coordinates $f\left( {r,\theta } \right)$, you can follow these steps:

$F\left( {\rho ,\varphi } \right)\mathop  \leftrightarrow \limits^{FF{T_\varphi }} {F_n}\left( \rho  \right)\mathop  \leftrightarrow \limits^{{H_n}} {f_n}\left( r \right)\mathop  \leftrightarrow \limits^{IFF{T_\theta }} f\left( {r,\theta } \right)$

Meaning that you need to take two FFTs and one Hankel Transform. FFT is already implemented in `numpy.fft.fft`. However, hankel transform is not implemented. Also, in general, there is no package to handle this type of data natively in Python. This package serves as a toolkit to reconstruct polar MR images using PFT and handle the images after that.

Based on:

    Golshani, S., & Nasiraei‐Moghaddam, A. (2017). Efficient radial tagging CMR exam: A coherent k‐space reading and image reconstruction approach. Magnetic resonance in medicine, 77(4), 1459-1472. https://doi.org/10.1002/mrm.26219

## Dependencies

```"numpy>=2.1"
"matplotlib>=2.0",
"joblib>=1.5",
"opencv-python>=4.12",
"platformdirs>=4.4",
"scipy>=1.16"
```

## Installing

Optional: Create a conda environment
```
conda create -n PyPFT python=3.13
```

Then, once the environment is activated, install it using pip

```
pip install pypft
```

You can run the tests when you are in the directory by running this command:

```
run-tests
```

You need to have the test files in order to run `test_*_sample` functions. However, you can run the random ones without test files.