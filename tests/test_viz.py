"""Tests for the visualization layer (``pypft.viz``).

``plot_signal`` is exercised for every domain's (magnitude, phase) return shape and
its own validation; ``render_cartesian`` similarly for its return shape and
validation. ``forward_pft_traced``/``inverse_pft_traced`` are checked against
``pypft.transform.forward_pft``/``inverse_pft`` directly, since they are meant to be
bit-identical (both walk the same ``pypft.domains`` chain). The figure-lifecycle
tests exist because ``pytest.ini_options: filterwarnings = ["error"]`` turns
matplotlib's own "too many open figures" warning into a test failure --
``PFTTrace.close()`` must actually return ``plt.get_fignums()`` to its prior length,
not merely close *some* figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypft.domains import (
    Domain,
    FrequencyPolarSignal,
    SpaceHarmonicSignal,
    SpacePolarSignal,
)
from pypft.grid import PolarGrid
from pypft.transform import forward_pft, inverse_pft
from pypft.viz import (
    PFTTrace,
    forward_pft_traced,
    inverse_pft_traced,
    plot_signal,
    render_cartesian,
)

_R = 40.0
_N_RADIAL = 32
_N_ANGULAR = 15


@pytest.fixture(autouse=True)
def _close_all_figures_after_test():
    """Close every figure a test leaves open, so the suite's own figure count never grows.

    A safety net on top of each test's own explicit ``close``/assertion calls --
    without it, a large suite of viz tests would eventually trip matplotlib's "more
    than 20 figures" warning, which ``filterwarnings = ["error"]`` escalates to a
    test failure.
    """
    yield
    plt.close("all")


def _grid() -> PolarGrid:
    return PolarGrid(n_radial=_N_RADIAL, n_angular=_N_ANGULAR, R=_R)


def _space_polar_signal() -> SpacePolarSignal:
    grid = _grid()
    return SpacePolarSignal(values=np.exp(-(grid.r.T**2)), grid=grid)


def _space_harmonic_signal() -> SpaceHarmonicSignal:
    grid = _grid()
    return SpaceHarmonicSignal(values=np.exp(-(grid.r.T**2)), grid=grid)


def _frequency_polar_signal() -> FrequencyPolarSignal:
    grid = _grid()
    return FrequencyPolarSignal(
        values=np.pi * np.exp(-(grid.rho.T**2) / 4.0), grid=grid
    )


def _varying_phase_frequency_polar_signal() -> FrequencyPolarSignal:
    """A ``FrequencyPolarSignal`` whose phase actually varies, unlike the fixture above.

    ``_frequency_polar_signal`` (like ``_space_polar_signal``) is real-valued, so its
    phase is exactly ``0.0`` everywhere -- useful for the degenerate-phase regression
    case, but useless for checking that a *non*-degenerate phase still gets the fixed
    ``[-pi, pi]`` color range rather than one auto-scaled to its own narrower spread.
    """
    grid = _grid()
    magnitude = np.exp(-(grid.rho.T**2) / 4.0)
    phase = np.linspace(-np.pi, np.pi, magnitude.size).reshape(magnitude.shape)
    return FrequencyPolarSignal(values=magnitude * np.exp(1j * phase), grid=grid)


# ======================================================================================
# plot_signal
# ======================================================================================


def test_plot_signal_space_harmonic_domain_returns_two_axes():
    """``Domain.SPACE_HARMONIC`` is assumed complex too, like every other domain."""
    magnitude_ax, phase_ax = plot_signal(signal=_space_harmonic_signal())
    assert isinstance(magnitude_ax, Axes)
    assert isinstance(phase_ax, Axes)
    assert magnitude_ax is not phase_ax


def test_plot_signal_space_polar_domain_returns_two_axes():
    """``Domain.SPACE_POLAR`` is assumed complex, rendered as a (magnitude, phase) pair."""
    magnitude_ax, phase_ax = plot_signal(signal=_space_polar_signal())
    assert isinstance(magnitude_ax, Axes)
    assert isinstance(phase_ax, Axes)
    assert magnitude_ax is not phase_ax


def test_plot_signal_frequency_domain_returns_two_axes():
    """A frequency-domain signal renders as a (magnitude, phase) pair."""
    magnitude_ax, phase_ax = plot_signal(signal=_frequency_polar_signal())
    assert isinstance(magnitude_ax, Axes)
    assert isinstance(phase_ax, Axes)
    assert magnitude_ax is not phase_ax


def test_plot_signal_space_polar_magnitude_uses_grayscale_colormap():
    """``Domain.SPACE_POLAR`` is a photographic image, so its magnitude is grayscale."""
    magnitude_ax, phase_ax = plot_signal(signal=_space_polar_signal())
    assert magnitude_ax.images[0].get_cmap().name == "gray"
    assert phase_ax.images[0].get_cmap().name != "gray"


def test_plot_signal_frequency_polar_magnitude_keeps_default_colormap():
    """A non-``SPACE_POLAR`` domain's magnitude keeps ``matplotlib``'s default colormap."""
    magnitude_ax, _ = plot_signal(signal=_frequency_polar_signal())
    assert magnitude_ax.images[0].get_cmap().name != "gray"


def test_plot_signal_phase_color_range_is_fixed_for_constant_phase():
    """A real-valued signal's (phase=0 everywhere) phase panel still spans [-pi, pi].

    See ``DESIGN_NOTES.md``, "Visualization: phase color range is pinned to
    ``[-pi, pi]``," for why an unpinned range would collapse to
    ``get_clim() == (0.0, 0.0)`` here instead.
    """
    _, phase_ax = plot_signal(signal=_space_polar_signal())
    assert phase_ax.images[0].get_clim() == (-np.pi, np.pi)


def test_plot_signal_phase_color_range_is_fixed_for_varying_phase():
    """A signal with genuinely varying phase still gets the same fixed [-pi, pi] range.

    The color range stays pinned to ``[-pi, pi]`` unconditionally -- not merely
    because this signal's own phase happens to already span close to that range
    (unlike the constant-phase case above).
    """
    _, phase_ax = plot_signal(signal=_varying_phase_frequency_polar_signal())
    assert phase_ax.images[0].get_clim() == (-np.pi, np.pi)


def test_plot_signal_reuses_given_axes_for_a_frequency_domain_signal():
    """A caller-supplied ``(magnitude_ax, phase_ax)`` pair is used directly."""
    _, (given_magnitude_ax, given_phase_ax) = plt.subplots(nrows=1, ncols=2)
    magnitude_ax, phase_ax = plot_signal(
        signal=_frequency_polar_signal(), ax=(given_magnitude_ax, given_phase_ax)
    )
    assert magnitude_ax is given_magnitude_ax
    assert phase_ax is given_phase_ax


def test_plot_signal_rejects_a_non_signal():
    """``plot_signal`` type-validates ``signal``."""
    with pytest.raises(TypeError):
        plot_signal(signal=np.zeros((_N_RADIAL, _N_ANGULAR)))  # type: ignore[arg-type]


def test_plot_signal_rejects_a_3d_signal():
    """``plot_signal`` only supports 2-D (unbatched) signals."""
    grid = _grid()
    values = np.zeros((_N_RADIAL, _N_ANGULAR, 3), dtype=complex)
    signal = SpacePolarSignal(values=values, grid=grid)
    with pytest.raises(ValueError):
        plot_signal(signal=signal)


def test_plot_signal_rejects_a_non_axes_ax():
    """``plot_signal`` type-validates a given ``ax``."""
    with pytest.raises(TypeError):
        plot_signal(signal=_space_polar_signal(), ax="not an axes")  # type: ignore[arg-type]


def test_plot_signal_rejects_a_wrong_length_tuple_ax():
    """``ax`` must be a 2-tuple, not just any tuple.

    Guards ``plot_signal``'s own type check, which must run before unpacking
    ``ax`` -- an unguarded ``magnitude_ax, phase_ax = ax`` on a wrong-length
    tuple raises ``ValueError`` instead of ``TypeError``, breaking
    ``plot_signal``'s own documented type-validation contract.
    """
    _, given_ax = plt.subplots()
    with pytest.raises(TypeError):
        plot_signal(signal=_space_polar_signal(), ax=(given_ax,))  # type: ignore[arg-type]


def test_base_signal_plot_delegates_to_plot_signal():
    """``BaseSignal.plot`` is a thin delegate to ``pypft.viz.plot_signal``."""
    signal = _space_harmonic_signal()
    magnitude_ax, phase_ax = signal.plot()
    assert isinstance(magnitude_ax, Axes)
    assert isinstance(phase_ax, Axes)


# ======================================================================================
# render_cartesian
# ======================================================================================


def test_render_cartesian_returns_the_given_axes():
    """``render_cartesian`` draws onto (and returns) a caller-supplied ``ax``."""
    _, given_ax = plt.subplots()
    returned_ax = render_cartesian(
        signal=_space_polar_signal(), height=32, width=32, ax=given_ax
    )
    assert returned_ax is given_ax


def test_render_cartesian_creates_an_axes_when_none_given():
    """``render_cartesian`` creates its own ``Axes`` if ``ax`` is not given."""
    ax = render_cartesian(signal=_frequency_polar_signal(), height=16, width=16)
    assert isinstance(ax, Axes)


def test_render_cartesian_space_polar_uses_grayscale_colormap():
    """``render_cartesian`` renders ``Domain.SPACE_POLAR`` in grayscale, like a photo."""
    ax = render_cartesian(signal=_space_polar_signal(), height=16, width=16)
    assert ax.images[0].get_cmap().name == "gray"


def test_render_cartesian_frequency_polar_keeps_default_colormap():
    """``render_cartesian`` leaves ``Domain.FREQUENCY_POLAR`` at the default colormap."""
    ax = render_cartesian(signal=_frequency_polar_signal(), height=16, width=16)
    assert ax.images[0].get_cmap().name != "gray"


def test_render_cartesian_rejects_a_harmonic_domain_signal():
    """Only the two ``POLAR`` domains have a physical angle axis to interpolate onto."""
    grid = _grid()
    values = np.zeros((_N_RADIAL, _N_ANGULAR), dtype=complex)
    signal = SpaceHarmonicSignal(values=values, grid=grid)
    with pytest.raises(ValueError):
        render_cartesian(signal=signal, height=16, width=16)


def test_render_cartesian_rejects_a_non_positive_height():
    """``render_cartesian`` validates ``height``/``width`` are positive."""
    with pytest.raises(ValueError):
        render_cartesian(signal=_space_polar_signal(), height=0, width=16)


def test_render_cartesian_orientation_matches_image_convention():
    """A signal concentrated at ``theta=+pi/2`` renders in the lower half.

    ``grid.theta`` follows ``pypft.geometry``'s convention -- measured
    directly on image coordinates, so increasing angle rotates toward
    increasing row (downward on screen), not toward standard math "up".
    ``theta=+pi/2`` is therefore the *bottom* half of the original image, and
    a correctly oriented ``render_cartesian`` output must show it in the
    physical lower half (``y < 0``, drawn at the bottom by ``origin="lower"``)
    -- not the upper half, which is what an unnegated ``sin(theta)`` produces.

    """
    grid = PolarGrid(n_radial=64, n_angular=64, R=40.0)
    theta0 = np.pi / 2
    angular_distance = np.abs((grid.theta - theta0 + np.pi) % (2 * np.pi) - np.pi)
    near_theta0 = angular_distance < (2 * np.pi / grid.n_angular)
    values = np.zeros((grid.n_radial, grid.n_angular))
    values[grid.n_radial // 2 :, near_theta0] = 1.0  # a bright wedge near theta0
    signal = SpacePolarSignal(values=values, grid=grid)

    ax = render_cartesian(signal=signal, height=64, width=64)

    # np.nanmean would warn ("Mean of empty slice") on any all-NaN row outside
    # the interpolated disc, which filterwarnings = ["error"] escalates to a
    # failure -- nan_to_num + a plain sum sidesteps that entirely.
    image = np.nan_to_num(np.asarray(ax.images[0].get_array()), nan=0.0)
    yi = np.linspace(-grid.R, grid.R, 64)
    row_brightness = image.sum(axis=1)
    assert row_brightness.sum() > 0
    mean_y = np.average(yi, weights=row_brightness)
    assert mean_y < 0


# ======================================================================================
# forward_pft_traced / inverse_pft_traced
# ======================================================================================


def test_forward_pft_traced_values_match_forward_pft():
    """A traced forward PFT is bit-identical to ``forward_pft`` -- same chain, same call."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))

    trace = forward_pft_traced(f=f, grid=grid)

    expected = forward_pft(f=f, grid=grid)
    np.testing.assert_array_equal(trace.values, expected)
    assert [signal.domain for signal in trace.signals] == [
        Domain.SPACE_POLAR,
        Domain.SPACE_HARMONIC,
        Domain.FREQUENCY_HARMONIC,
        Domain.FREQUENCY_POLAR,
    ]
    assert trace.figures == ()


def test_inverse_pft_traced_values_match_inverse_pft():
    """A traced inverse PFT is bit-identical to ``inverse_pft``."""
    grid = _grid()
    F = np.pi * np.exp(-(grid.rho.T**2) / 4.0)

    trace = inverse_pft_traced(F=F, grid=grid)

    expected = inverse_pft(F=F, grid=grid)
    np.testing.assert_array_equal(trace.values, expected)
    assert [signal.domain for signal in trace.signals] == [
        Domain.FREQUENCY_POLAR,
        Domain.FREQUENCY_HARMONIC,
        Domain.SPACE_HARMONIC,
        Domain.SPACE_POLAR,
    ]


def test_forward_pft_traced_visualize_steps_creates_one_figure_per_domain():
    """``visualize_steps`` records exactly one figure per one of the 4 domains."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))

    trace = forward_pft_traced(f=f, grid=grid, visualize_steps=True)

    assert len(trace.figures) == 4
    for figure in trace.figures:
        assert isinstance(figure, Figure)
    trace.close()


def test_forward_pft_traced_visualize_pipeline_creates_one_mosaic_figure():
    """``visualize_pipeline`` records exactly one holistic mosaic figure."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))

    trace = forward_pft_traced(f=f, grid=grid, visualize_pipeline=True)

    assert len(trace.figures) == 1
    assert isinstance(trace.figures[0], Figure)
    trace.close()


def test_forward_pft_traced_both_visualizations_creates_five_figures():
    """Both keywords together record 4 step figures plus 1 mosaic figure."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))

    trace = forward_pft_traced(
        f=f, grid=grid, visualize_steps=True, visualize_pipeline=True
    )

    assert len(trace.figures) == 5
    trace.close()


def test_pfttrace_close_restores_the_prior_figure_count():
    """``PFTTrace.close`` returns ``plt.get_fignums()`` to its length before tracing."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))
    before = len(plt.get_fignums())

    trace = forward_pft_traced(
        f=f, grid=grid, visualize_steps=True, visualize_pipeline=True
    )
    assert len(plt.get_fignums()) > before

    trace.close()

    assert len(plt.get_fignums()) == before


def test_pfttrace_rejects_a_non_figure_in_figures():
    """``PFTTrace.__post_init__`` type-validates every element of ``figures``."""
    with pytest.raises(TypeError):
        PFTTrace(values=np.zeros((1,)), signals=(), figures=("not a figure",))  # type: ignore[arg-type]


def test_pfttrace_rejects_figures_and_figure_labels_of_different_lengths():
    """``PFTTrace.__post_init__`` requires one label per figure."""
    figure = plt.figure()
    try:
        with pytest.raises(ValueError):
            PFTTrace(
                values=np.zeros((1,)),
                signals=(),
                figures=(figure,),
                figure_labels=(),
            )
    finally:
        plt.close(figure)


# ======================================================================================
# PFTTrace.save
# ======================================================================================


def test_pfttrace_save_writes_one_enumerated_file_per_figure(tmp_path):
    """``save`` writes ``"<index>_<label>.png"`` for every figure, in order."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))
    trace = forward_pft_traced(f=f, grid=grid, visualize_steps=True)

    written = trace.save(directory=tmp_path)

    assert written == (
        tmp_path / "00_step_space_polar.png",
        tmp_path / "01_step_space_harmonic.png",
        tmp_path / "02_step_frequency_harmonic.png",
        tmp_path / "03_step_frequency_polar.png",
    )
    for path in written:
        assert path.is_file()
    trace.close()


def test_pfttrace_save_creates_missing_parent_directories(tmp_path):
    """``save`` creates ``directory`` (and any missing parents) if needed."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))
    trace = forward_pft_traced(f=f, grid=grid, visualize_pipeline=True)
    nested = tmp_path / "a" / "b" / "c"

    written = trace.save(directory=nested)

    assert nested.is_dir()
    assert written[0].is_file()
    trace.close()


def test_pfttrace_save_rejects_a_non_path_directory():
    """``save`` type-validates ``directory``."""
    grid = _grid()
    f = np.exp(-(grid.r.T**2))
    trace = forward_pft_traced(f=f, grid=grid, visualize_pipeline=True)
    try:
        with pytest.raises(TypeError):
            trace.save(directory="not a path")  # type: ignore[arg-type]
    finally:
        trace.close()
