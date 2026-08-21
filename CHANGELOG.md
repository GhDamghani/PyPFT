# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `pypft.domains`: `Domain`, `BaseSignal`, and its four subclasses
  (`SpacePolarSignal`, `SpaceHarmonicSignal`, `FrequencyHarmonicSignal`,
  `FrequencyPolarSignal`) -- a typed, thin shell over `forward_pft`/
  `inverse_pft` that names where a polar array sits along the PFT's domain
  chain and walks between those points one verified step at a time, either
  by hand (destination-named methods, so an illegal step is a `pyright`
  error) or dynamically (`BaseSignal.to`).
- `notebooks/05_domains.ipynb`: the typed-domains tutorial.
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
