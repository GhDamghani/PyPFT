User Guide
==========

A tour of PyPFT's public API, one layer at a time. The tutorial notebooks
(:doc:`tutorials`) cover each layer in depth; this page is the short version.

The discrete Hankel transform
-----------------------------

.. code-block:: python

   import numpy as np
   import pypft

   n, R = 0, 8.0
   r, rho = pypft.sample_points(n, size=64, R=R)

   f = np.exp(-(r**2) / 2)
   F = pypft.hankel_transform(f, n, R)
   f_reconstructed = pypft.inverse_hankel_transform(F, n, R)

The angular discrete Fourier transform
--------------------------------------

``pypft.dft.angular_dft``/``pypft.dft.inverse_angular_dft`` compute the centered angular
DFT/IDFT -- the two angular steps of the polar Fourier transform, kept centered the same
way every other PyPFT array is (index ``n_angular // 2`` holds harmonic ``0``). This is a
lower-level building block of ``pypft.forward_pft``/``pypft.inverse_pft`` rather than a
typical end-user entry point, so it is not re-exported from the top-level ``pypft``
package. ``pypft.dft.harmonics`` gives the centered harmonic-index range for a given
angular sample count (valid for either an odd or an even count), and
``pypft.dft.AngularParity`` names that parity:

.. code-block:: python

   import numpy as np
   from pypft.dft import angular_dft, inverse_angular_dft

   x = np.random.default_rng(0).standard_normal(16)
   X = angular_dft(x)
   x_reconstructed = inverse_angular_dft(X)

Cartesian and polar images
--------------------------

``pypft.cartesian_to_polar``/``pypft.polar_to_cartesian`` resample an ordinary image onto
(and back off of) a *uniform* polar grid. This is not the transform's own
(order-dependent, non-uniform) sampling grid -- it is the natural first illustration of
what "polar" means for an image:

.. code-block:: python

   polar = pypft.cartesian_to_polar(image, n_radial=128, n_angular=96)
   reconstructed = pypft.polar_to_cartesian(polar, height, width)

The returned array follows PyPFT's own ``(radial, angular)`` axis layout (``pypft.Axis``),
with a centered angular axis: index ``n_angular // 2`` holds angle ``0``.

The transform's own sampling grid
---------------------------------

``pypft.PolarGrid`` is the transform's *actual* sampling grid -- uniform in angle,
non-uniform in radius. Row ``i`` of ``grid.r`` is the spoke at angle ``grid.theta[i]``,
and its radii come from the zeros of the Bessel function of order
``abs(grid.harmonics[i])``, the spoke's own angular sample index. The transform
identifies that index with the harmonic order (Mathematics Part I, Eq. 18), which is
what makes it exactly invertible:

.. code-block:: python

   grid = pypft.PolarGrid(n_radial=383, n_angular=15, R=40.0)
   grid.r      # (n_angular, n_radial) space-domain radii
   grid.theta  # (n_angular,) centered angles, shared by both domains

``pypft.sample_cartesian`` resamples an ordinary image directly onto a grid's own points:

.. code-block:: python

   polar = pypft.sample_cartesian(image, grid)

``pypft.resample_uniform_polar`` puts uniform polar data on the grid instead: a
``cartesian_to_polar`` result, or an acquisition with equally spaced samples along each
spoke. It interpolates each spoke along the radius with a cubic spline onto ``grid.r``
(and across the spokes too, periodically, when the data has a different number of
spokes), and returns ``forward_pft``'s ``(radial, angular[, batch])`` input directly.
Its ``radius`` is the radius the uniform samples cover, in ``cartesian_to_polar``'s
convention: sample ``k`` of ``n`` sits at ``k * radius / n``, so for a
``cartesian_to_polar`` result ``radius`` is half the smaller side of the image:

.. code-block:: python

   uniform = pypft.cartesian_to_polar(image, n_radial=256, n_angular=grid.n_angular)
   f = pypft.resample_uniform_polar(uniform, grid, radius=min(image.shape) / 2)
   F = pypft.forward_pft(f, grid)

``pypft.check_adequacy``/``pypft.check_nyquist_adequacy`` warn (never raise) when a grid's
``n_radial`` is too small for its ``n_angular``, or violates the discrete Hankel
transform's own Nyquist condition, respectively -- both are easy mistakes to make
silently, since neither failure mode raises an error on its own:

.. code-block:: python

   pypft.check_adequacy(grid)  # silent for this grid
   pypft.check_adequacy(pypft.PolarGrid(n_radial=383, n_angular=64, R=40.0))  # warns

``check_adequacy`` predicts the forward transform's relative L2 error from a fit of a
centered Gaussian's measured error to ``(n_angular, n_radial)`` at ``R = 40``, and warns
when the prediction is worse than on Yao & Baddour's worked example
(``n_radial=382, n_angular=15, R=40``). An off-center function, or an ``R`` small
relative to the object, can still give a much larger error (see "Which grid?" below), so
its silence is necessary, not sufficient, for an accurate approximation.

Which grid?
~~~~~~~~~~~

Three grids play a role, and they answer different questions:

- **The uniform polar grid** (``cartesian_to_polar``) is what your data is on: the same
  equally spaced radii on every spoke, i.e. true rings.
- **The transform's grid** (``PolarGrid``) is where the transform needs its samples if
  its output is meant to approximate the continuous Fourier transform. It has the same
  spokes for the same ``n_angular``, but different radii on each spoke, so a row of
  constant radial index is not a ring.
- **The bridge** between them is interpolation along each spoke, from the uniform radii
  onto ``grid.r``: ``pypft.resample_uniform_polar``. Measured, it gives the same
  transform as sampling on the grid directly.

So, can the PFT be applied to a uniformly polar-sampled image?

1. *As a discrete transform*, yes: ``forward_pft``/``inverse_pft`` apply to any
   ``(n_radial, n_angular)`` array and invert each other to rounding error, and every
   discrete rule (orthogonality, shift, convolution, Parseval) holds.
2. *As an approximation of the continuous 2-D Fourier transform*, only after
   interpolating it along each spoke onto ``grid.r`` with
   ``pypft.resample_uniform_polar``. Fed as-is, a uniform array is read
   as a function warped differently along each spoke.
3. *Even on the grid*, the approximation carries the error of identifying a spoke with a
   harmonic. It is small only for content that varies slowly across the per-spoke radius
   offsets; report the relative L2 error, not only the average dB error, to see it.

``DESIGN_NOTES.md``, "Grid: the spatial row index is a spoke, and the transform
identifies it with a harmonic," has the measurements, and the
``02_sampling_grids`` tutorial shows them.

The full PFT/IPFT pipeline
--------------------------

``pypft.forward_pft``/``pypft.inverse_pft`` chain the angular DFT/IDFT with a
per-harmonic, ``R``-scaled discrete Hankel transform. Both take a ``pypft.PolarGrid`` and
an ``(n_radial, n_angular)`` array in PyPFT's own layout (``pypft.Axis``) -- **not**
``PolarGrid.r``'s/``pypft.sample_cartesian``'s own ``(n_angular, n_radial)`` layout, so
transpose a ``sample_cartesian`` result first:

.. code-block:: python

   grid = pypft.PolarGrid(n_radial=382, n_angular=15, R=40.0)
   f = np.exp(-(grid.r.T**2))  # a radially symmetric Gaussian, on grid.r.T
   F = pypft.forward_pft(f, grid)  # the frequency-domain samples F(rho, phi)
   f_reconstructed = pypft.inverse_pft(F, grid)

This is the **exact path**, and the default everywhere: square and exactly invertible for
any array, with every discrete rule holding exactly, but as an approximation of the
continuous transform it carries the spoke-to-harmonic identification error of "Which
grid?" above.

The ring-consistent route
~~~~~~~~~~~~~~~~~~~~~~~~~

``pypft.forward_pft_ring``/``pypft.inverse_pft_ring`` are the alternative when the output
must approximate the continuous Fourier transform. The route computes every harmonic from
samples on true rings at its own radii, applies the same per-harmonic discrete Hankel
transform, and evaluates every harmonic at each output spoke's own radii through the
transform's Fourier-Bessel expansion, so no spoke is identified with a harmonic on either
side. Its input is a Cartesian image, or, given ``radius``, uniform polar data in
``resample_uniform_polar``'s convention; its output is on ``forward_pft``'s own frequency
grid:

.. code-block:: python

   F = pypft.forward_pft_ring(image, grid)  # a Cartesian image, R in pixels
   F = pypft.forward_pft_ring(uniform, grid, radius=40.0)  # uniform polar data
   f = pypft.inverse_pft_ring(F, grid)  # an approximation, not an exact inverse

Its pieces are public too: ``pypft.sample_harmonics_cartesian``/
``pypft.sample_harmonics_uniform_polar`` return a spatial-harmonic array (the values of a
``pypft.PolarSpatialHarmonicSignal``, entering the domain chain one step in),
``pypft.evaluate_frequency`` replaces the chain's last step, and ``pypft.evaluate_space`` is
its space-side mirror:

.. code-block:: python

   harmonics = pypft.sample_harmonics_cartesian(image, grid, n_quadrature=1024)
   signal = pypft.PolarSpatialHarmonicSignal(values=harmonics, grid=grid)
   F = pypft.evaluate_frequency(signal.to_polar_frequency_harmonic().values, grid)

Use the route for accuracy, the exact path for exact invertibility. The route removes the
identification error; what remains is set by its input stage. Where the grid's harmonics
hold the function, uniform polar input measures ``1e-7`` to ``1e-5`` relative L2 error
(the radial spline), where the exact path measures ``0.3`` to ``2``, and a smooth
Cartesian image a few ``1e-3`` (the pixel interpolation), where the exact path is off by
more than the transform's own size; otherwise the harmonics beyond the grid's range set
the error. It is not
exactly invertible, though: its inverse is accurate only when the frequency grid's central
gap holds little of the transform (large ``R`` relative to the object, few spokes), and its
output stage costs ``n_angular**2 / 4`` Bessel matrices of ``n_radial**2`` values each.
Both directions support a trailing batch axis and ``LimitKind.SPACE_LIMITED`` grids only.
``DESIGN_NOTES.md``, "PFT: the exact path and the ring-consistent route," has the
measurements, and the ``05_pft_and_ipft`` tutorial's "Two paths" section compares the
two.

Typed domains and legal moves
-----------------------------

``pypft.Domain``/``pypft.BaseSignal`` add nothing numerical on top of
``pypft.forward_pft``/``pypft.inverse_pft``. They add a *typed* way to name where a polar
array sits along the chain, and to walk between those points one step at a time::

   POLAR_SPATIAL --DFT--> POLAR_SPATIAL_HARMONIC
                 --DHT--> POLAR_FREQUENCY_HARMONIC
                 --IDFT--> POLAR_FREQUENCY

Every ``Domain`` member starts with ``POLAR``, since every point on the chain is sampled
on the same ``pypft.PolarGrid``. The word after it (``SPATIAL``/``FREQUENCY``) is the
radial coordinate, changed only by the discrete Hankel transform; a trailing ``HARMONIC``
marks the angular coordinate as a harmonic order rather than a physical angle, changed
only by the angular DFT/IDFT. ``pypft.BaseSignal``'s four subclasses
(``PolarSpatialSignal``, ``PolarSpatialHarmonicSignal``,
``PolarFrequencyHarmonicSignal``, ``PolarFrequencySignal``) -- one per ``Domain`` member
-- wrap a ``(values, grid)`` pair with the domain it currently occupies, and know only
the neighbouring domains they may legally step to. Each step method is named after the
domain it moves *into*, so a hand-written chain reads as the chain itself:

.. code-block:: python

   signal = pypft.PolarSpatialSignal(f, grid)
   harmonic_signal = signal.to_polar_spatial_harmonic()  # a PolarSpatialHarmonicSignal
   by_hand = (  # a PolarFrequencySignal
       signal.to_polar_spatial_harmonic()
       .to_polar_frequency_harmonic()
       .to_polar_frequency()
   )

   # `to` walks the same chain dynamically, to any target domain:
   walked = signal.to(pypft.Domain.POLAR_FREQUENCY)

``by_hand``/``walked`` both match ``pypft.forward_pft(f, grid)`` exactly, since each step
method is a thin wrapper around the same calls ``forward_pft`` itself makes.

3-D batches
-----------

``pypft.forward_pft``/``pypft.inverse_pft`` (and every ``pypft.BaseSignal`` step method)
also accept a 3-D ``(n_radial, n_angular, batch)`` array -- the same two polar axes, plus
one trailing batch axis (``pypft.Axis.BATCH``, ``pypft.DEFAULT_BATCH_AXIS``). Batching
changes only how many signals one call transforms, never what it computes:

.. code-block:: python

   widths = np.array([0.5, 1.0, 2.0, 4.0])
   f_batch = np.exp(-widths * grid.r.T[..., np.newaxis] ** 2)  # (n_radial, n_angular, 4)
   F_batch = pypft.forward_pft(f_batch, grid)  # every width transformed in one call

The batch axis is always last -- passing one anywhere else, or on a plain 2-D array,
raises immediately.

Visualization
-------------

``pypft.plot_signal``/``pypft.BaseSignal.plot`` render a signal as a ``matplotlib``
``(magnitude, phase)`` pair of ``Axes``, for every ``pypft.Domain`` member alike. The
magnitude is gamma-enhanced, and ``POLAR_SPATIAL``'s magnitude is drawn in grayscale,
since it is literally an image. ``pypft.render_cartesian`` interpolates a
``POLAR_SPATIAL``/``POLAR_FREQUENCY`` signal's own non-uniform sample points onto an
ordinary Cartesian grid -- for display only; its output must never be fed back into
``pypft.forward_pft``/``pypft.inverse_pft``:

.. code-block:: python

   grid = pypft.PolarGrid(n_radial=96, n_angular=31, R=40.0)
   signal = pypft.PolarSpatialSignal(np.exp(-(grid.r.T**2)), grid)

   signal.plot()
   signal.to_polar_spatial_harmonic().plot()
   signal.to(pypft.Domain.POLAR_FREQUENCY).plot()
   pypft.render_cartesian(signal, height=256, width=256)

``pypft.forward_pft_traced``/``pypft.inverse_pft_traced`` run the same pipeline as
``pypft.forward_pft``/``pypft.inverse_pft``, but return a ``pypft.PFTTrace`` recording
every domain (and, optionally, every figure) along the way -- ``visualize_steps=True``
renders one figure per domain, and ``visualize_pipeline=True`` renders one mosaic of all
of them:

.. code-block:: python

   from pathlib import Path

   trace = pypft.forward_pft_traced(
       f, grid, visualize_steps=True, visualize_pipeline=True
   )
   trace.values  # identical to pypft.forward_pft(f, grid)
   len(trace.figures)  # 4 step figures + 1 pipeline mosaic
   trace.save(Path("out"))  # writes every figure as an enumerated, domain-named PNG
   trace.close()  # frees every figure the trace created

``PFTTrace.save`` is the one place PyPFT ever writes to disk -- only when called
explicitly, never as a side effect of tracing itself.

Citing PyPFT
------------

The repository's `CITATION.cff <https://github.com/GhDamghani/PyPFT/blob/main/CITATION.cff>`_
holds the metadata for citing PyPFT itself, and lists the papers behind its math as
references. Each tutorial notebook cites the results it states with numbered links, such
as ``[1] (Eq. 41)``, to a References section at its end.
