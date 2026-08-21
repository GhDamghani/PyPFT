# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Analytical-property tests for both transforms: the DHT's negative-order
  sign relation and its shift/modulation/multiplication/convolution rules
  (`tests/dht/test_kernel_properties.py`), and the PFT's kernel
  orthogonality, complex-exponential/delta pair, generalized shift and its
  derived rules, rotation equivariance, linearity, DC term, and twisted
  conjugate symmetry (`tests/test_transform_properties.py`).
- `pypft._kernel.kernel_matrix`: an explicit, from-scratch `O(N**4)`
  operator reproducing `forward_pft`/`inverse_pft` exactly (PeerJ CS Part
  II, Eqs. 1-3) -- a from-first-principles oracle for
  `tests/test_kernel.py`, not part of the public API.
- `notebooks/04_transform_properties.ipynb`: a tour of both transforms'
  analytical properties, citing Baddour's DHT chapter and the first part of
  the two-part polar-coordinates paper.
- `pypft.forward_pft`/`pypft.inverse_pft`: the full polar Fourier transform
  (PFT) / inverse PFT (IPFT) pipeline, chaining the angular DFT/IDFT with a
  per-harmonic, `R`-scaled discrete Hankel transform on a `pypft.PolarGrid`.
  Reproduces Yao & Baddour's own published forward/inverse dB-error figures
  exactly (PeerJ CS Part II, Appendix A-5/A-6).
- `pypft.transform.scaled_hankel`/`pypft.transform.Direction`: the
  per-harmonic sign- and scale-corrected Hankel transform step underlying
  `forward_pft`/`inverse_pft`.
- `NumpyValidator.value1_shape_matches_value2` in
  `pypft.utils.validators`, for validating a whole array shape against a
  reference in one call.
- `notebooks/03_pft_and_ipft.ipynb`: the PFT/IPFT tutorial, reproducing the
  paper's own dB-error figure.
