Conventions
===========

The current scaffold freezes only structural conventions.

- Spatial-domain arrays use the axis order ``(radial, angular)``.
- Frequency-domain arrays use the axis order ``(radial, angular)``.
- Batched execution uses a leading batch axis.
- Arrays are normalized to ``complex128`` at the API boundary.
- Angular transforms are delegated to ``scipy.fft`` wrappers.

The following mathematical conventions remain intentionally unspecified until the DHT implementation work begins:

- radial sampling semantics
- angular sampling semantics
- forward and backward normalization
- phase conventions
- exact forward and inverse DHT scaling rules

Those choices must be derived from the papers stored under ``.local_files/sources``.
