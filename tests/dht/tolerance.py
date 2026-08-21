"""A documented tolerance model for the DHT kernel's order-dependent error.

Baddour's ``Y^{nN}`` kernel (baddour2019.md, Eq. 39) is only approximately
self-inverse in floating point, and that residual -- along with every other
error that traces back to it (round-trip application, Parseval, forward/inverse
round-trip) -- grows with the transform's order ``n``, because the kernel
evaluates ``J_n`` at arguments that shrink relative to ``n`` as the order rises.
A single flat ``RTOL``/``ATOL`` (as used elsewhere in this suite, see
``tests/dht/conftest.py``) is therefore wrong once high orders are exercised:
it is far looser than necessary at ``n=0`` and already too tight
by order ~24 at ``size=64`` -- which is exactly the gap that let the (now
deleted) ``RecurrenceBesselDHT`` diverge unnoticed, since ``DHT_ORDERS`` used
to stop at 4. ``dht_tolerance`` replaces the flat bound for assertions whose
expected error is order-sensitive.

Fitted from ``NaiveDHT``/``CachedBesselDHT`` self-inverse residuals
(``max|Y @ Y - I|``), measured at ``size=64``:

    order   residual
    0       1.906e-09
    1       5.587e-09
    4       1.097e-07
    12      8.475e-07
    16      1.395e-06
    24      2.717e-06
    32      4.227e-06
    47      7.278e-06
    64      1.078e-05

This is an empirical fit, not a derived error bound: the residual grows
roughly as ``order**2 / size`` above the smallest orders, plus a floor for the
near-zero error at ``order=0``. ``ORDER_COEFFICIENT`` is chosen so the model
stays an upper bound, with roughly a 1.2x-3x margin, across every point above
-- comfortably inside the ~10x window ``test_tolerance.py`` checks this module
against, so a future regression (like the recurrence divergence) would still
be caught rather than silently absorbed by an over-loose model.
"""

#: Baseline error at order 0, where the residual is dominated by ordinary
#: floating-point round-off rather than order-dependent kernel degradation.
FLOOR = 5e-9

#: Coefficient of the ``order**2 / size`` growth term, fitted to stay an upper
#: bound (see module docstring) across every measured point.
ORDER_COEFFICIENT = 5e-7


def dht_tolerance(order: int, size: int) -> float:
    """Upper-bound the order-dependent DHT kernel error at a given size.

    :param order: The order of the discrete Hankel transform.
    :type order: int
    :param size: The length of the vectors the kernel transforms.
    :type size: int
    :returns: A tolerance expected to exceed the measured error by roughly
        1.2x-3x, per this module's docstring fit.
    :rtype: float

    """
    return FLOOR + ORDER_COEFFICIENT * order**2 / size
