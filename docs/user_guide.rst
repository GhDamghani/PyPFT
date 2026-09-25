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

``pypft.PolarGrid`` is the discrete Hankel transform's *actual* sampling grid --
order-dependent and non-uniform. Every angular row has its own radial sample positions,
tied to the zeros of a Bessel function of that row's harmonic order:

.. code-block:: python

   grid = pypft.PolarGrid(n_radial=383, n_angular=15, R=40.0)
   grid.r      # (n_angular, n_radial) space-domain radii
   grid.theta  # (n_angular,) centered angles, shared by both domains

``pypft.sample_cartesian`` resamples an ordinary image directly onto a grid's own points:

.. code-block:: python

   polar = pypft.sample_cartesian(image, grid)

``pypft.check_adequacy``/``pypft.check_nyquist_adequacy`` warn (never raise) when a grid's
``n_radial`` is too small for its ``n_angular``, or violates the discrete Hankel
transform's own Nyquist condition, respectively -- both are easy mistakes to make
silently, since neither failure mode raises an error on its own:

.. code-block:: python

   pypft.check_adequacy(grid)  # silent for this grid
   pypft.check_adequacy(pypft.PolarGrid(n_radial=383, n_angular=64, R=40.0))  # warns

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

Typed domains and legal moves
-----------------------------

``pypft.Domain``/``pypft.BaseSignal`` add nothing numerical on top of
``pypft.forward_pft``/``pypft.inverse_pft``. They add a *typed* way to name where a polar
array sits along the chain, and to walk between those points one step at a time::

   SPACE_POLAR --DFT--> SPACE_HARMONIC --DHT--> FREQUENCY_HARMONIC --IDFT--> FREQUENCY_POLAR

Each ``Domain`` member's first word (``SPACE``/``FREQUENCY``) is the radial coordinate,
changed only by the discrete Hankel transform; the second word (``POLAR``/``HARMONIC``) is
the angular coordinate, changed only by the angular DFT/IDFT. ``pypft.BaseSignal``'s four
subclasses (``SpacePolarSignal``, ``SpaceHarmonicSignal``, ``FrequencyHarmonicSignal``,
``FrequencyPolarSignal``) -- one per ``Domain`` member -- wrap a ``(values, grid)`` pair
with the domain it currently occupies, and know only the neighbouring domains they may
legally step to:

.. code-block:: python

   signal = pypft.SpacePolarSignal(f, grid)
   harmonic_signal = signal.to_harmonics()  # a SpaceHarmonicSignal
   by_hand = signal.to_harmonics().to_frequency().to_angles()  # a FrequencyPolarSignal

   # `to` walks the same chain dynamically, to any target domain:
   walked = signal.to(pypft.Domain.FREQUENCY_POLAR)

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
magnitude is gamma-enhanced, and ``SPACE_POLAR``'s magnitude is drawn in grayscale, since
it is literally an image. ``pypft.render_cartesian`` interpolates a
``SPACE_POLAR``/``FREQUENCY_POLAR`` signal's own non-uniform sample points onto an
ordinary Cartesian grid -- for display only; its output must never be fed back into
``pypft.forward_pft``/``pypft.inverse_pft``:

.. code-block:: python

   grid = pypft.PolarGrid(n_radial=96, n_angular=31, R=40.0)
   signal = pypft.SpacePolarSignal(np.exp(-(grid.r.T**2)), grid)

   signal.plot()
   signal.to_harmonics().plot()
   signal.to(pypft.Domain.FREQUENCY_POLAR).plot()
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

Citing a result
---------------

``pypft.Reference``/``pypft.cite``/``pypft.bibliography`` render the scientific sources
behind PyPFT's math:

.. code-block:: python

   pypft.cite(pypft.Reference.BADDOUR_2019_DHT)
   pypft.bibliography(pypft.Reference.BADDOUR_2019_DHT)
