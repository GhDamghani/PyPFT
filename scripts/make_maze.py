r"""Generate a reproducible circular maze directly on a PolarGrid, for a test fixture.

The maze is built in polar coordinates from the start: a central courtyard
surrounded by ``rings`` concentric rings of cells, each ring split into angular
sectors. Moving outward, a ring doubles the sector count of the ring inside it
whenever that keeps its cells roughly as wide as they are deep, up to a cap of
``n_angular // SAMPLES_PER_SECTOR`` sectors -- the most the grid can resolve
with one angular sample on every radial wall and the rest in the corridor
between. An iterative randomized depth-first search (the recursive
backtracker), driven only by ``numpy.random.default_rng(seed)``, carves a
perfect maze through the rings -- every cell reachable from every other by
exactly one path -- starting from an entrance in the outer wall, and one final
passage links the courtyard to the innermost ring. The same ``--seed`` gives
bit-identical output on every operating system.

The walls have a uniform physical thickness and are evaluated analytically at
the grid's own sample points (``pypft.PolarGrid.r``/``.theta``): circular walls
between rings, radial walls between sectors, and a round outer wall with one
entrance. Nothing is drawn on a Cartesian pixel grid first, so no
Cartesian-to-polar resampling blurs the circular walls: this is an image that
is naturally described, and best sampled, in polar coordinates. Every sector
count divides ``n_angular``, so every radial wall lies exactly on a sample
angle and even the thinnest one is hit by at least one angular sample.

The maze is sampled twice, once per kind of polar data:

- ``sample_maze_on_grid`` evaluates it at the ``PolarGrid``'s own points, every
  spoke at its own radii ``grid.r``. This is the exact path: the samples feed
  ``pypft.forward_pft`` directly, and they are the mainstream fixture.
- ``sample_maze_uniform_polar`` evaluates it on a uniform polar grid with the
  same spokes: the same equally spaced radii on every spoke, in
  ``pypft.cartesian_to_polar``'s convention. This is what uniform polar data
  looks like, and it goes through ``pypft.resample_uniform_polar`` before
  ``pypft.forward_pft``.

A third helper, ``sample_maze_rings``, computes the maze's angular harmonics on
each harmonic's own true rings, the input of the ring-consistent route
(``pypft.forward_pft_ring``). It writes no file: the route's input is generated
on the fly, so the exact path's ``maze_polar.tif`` stays the only maze fixture.

Four files are written to ``--output-dir``, next to each other:

- ``maze_polar.tif``: the fixture itself, the ``(n_radial, n_angular)`` polar
  array on the ``PolarGrid`` (walls ``0``, everything else ``255``) as an
  uncompressed 8-bit grayscale TIFF, ready to feed directly into
  ``pypft.PolarSpatialAngularSignal``/``pypft.forward_pft``.
- ``maze_uniform_polar.tif``: the same maze on the uniform polar grid with the
  same ``n_radial``, ``n_angular`` and radius ``R``, in the same format, to be
  resampled with ``pypft.resample_uniform_polar`` first.
- ``maze_cartesian.png``: ``pypft.render_cartesian``'s display of the
  ``PolarGrid`` signal, for visual reference.
- ``maze_source.png``: the same maze evaluated analytically on an ordinary
  square pixel grid, as an 8-bit grayscale PNG, for comparison against the
  rendering.

Usage:
    uv run python scripts/make_maze.py [--rings 8] [--inner-sectors 8] \
        [--seed 0] [--wall-thickness 0.25] [--n-radial 576] [--n-angular 48] \
        [--radius 1.0] [--maze-fraction 0.95] [--image-size 512] \
        [--output-dir tests/samples]

The committed fixtures in ``tests/samples/`` are exactly the output of the
defaults:

    uv run python scripts/make_maze.py

``pypft.check_adequacy`` runs on the requested grid and warns if ``n_radial``
is too small for ``n_angular``. The default grid warns: the maze is a sharp,
discontinuous display and regression image, not an accuracy reference, and its
grid is fixed by the committed fixture.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

import pypft
from pypft.dft import harmonics
from pypft.utils.validators import (
    FloatValidator,
    IntValidator,
    NumpyValidator,
    PathValidator,
)

# ========================================================================================
# Constants
# ========================================================================================

#: Rings of cells around the courtyard in the default maze.
DEFAULT_RINGS = 8

#: Sectors of the innermost ring in the default maze.
DEFAULT_INNER_SECTORS = 8

#: Seed of the default maze's random number generator.
DEFAULT_SEED = 0

#: Wall thickness of the default maze, as a fraction of one ring's width.
DEFAULT_WALL_THICKNESS = 0.25

#: The default ``PolarGrid``'s radial sample count.
DEFAULT_N_RADIAL = 576

#: The default ``PolarGrid``'s angular sample count.
DEFAULT_N_ANGULAR = 48

#: The default ``PolarGrid``'s space limit ``R``.
DEFAULT_RADIUS = 1.0

#: Radius of the default maze's outer wall, as a fraction of ``R`` -- a 5% margin
#: keeps the whole wall inside the grid.
DEFAULT_MAZE_FRACTION = 0.95

#: Side length, in pixels, of the two PNG files.
DEFAULT_IMAGE_SIZE = 512

#: Where the committed fixture lives: ``tests/samples/`` of this repository.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "samples"

#: File name of the polar-sampled fixture.
POLAR_FILENAME = "maze_polar.tif"

#: File name of the maze sampled on a uniform polar grid.
UNIFORM_POLAR_FILENAME = "maze_uniform_polar.tif"

#: File name of the ``render_cartesian`` display of the fixture.
CARTESIAN_FILENAME = "maze_cartesian.png"

#: File name of the maze evaluated on a square pixel grid.
SOURCE_FILENAME = "maze_source.png"

#: Angles evaluated around every ring by ``sample_maze_rings``: the thinnest radial
#: wall spans a few of them on the outermost ring.
DEFAULT_RING_QUADRATURE = 1024

#: Angular samples per sector of the outermost ring, at the most: one on the
#: radial wall at the sector's start, the rest in its corridor.
SAMPLES_PER_SECTOR = 3

#: Image intensity of a wall.
WALL_VALUE = 0.0

#: Image intensity of everything that is not a wall.
CORRIDOR_VALUE = 1.0

#: The courtyard, as a ``(ring, sector)`` cell: ring ``0``, one sector.
COURTYARD = (0, 0)

#: A ring's sector count doubles only if the doubled cells stay at least this wide
#: (arc length at the ring's middle radius) relative to the ring's own depth.
_MIN_CELL_ASPECT = 0.85

#: The fewest sectors a ring may have, so neighboring cells share one radial wall.
_MIN_SECTORS = 3

#: Largest value of an 8-bit grayscale pixel.
_UINT8_MAX = 255

#: Side length, in inches, of the square ``render_cartesian`` figure.
_FIGURE_SIZE = 6.0

#: One full turn, in radians.
_TURN = 2 * np.pi

# ========================================================================================
# Maze generation
# ========================================================================================


@dataclass(frozen=True)
class CircularMaze:
    """A perfect maze on concentric rings of cells around a courtyard.

    Ring ``k`` (``1 <= k <= len(sectors)``) spans radii ``k`` to ``k + 1`` in
    units of one ring's depth; the courtyard, ring ``0``, spans ``0`` to ``1``.
    Sector ``j`` of ring ``k`` spans angles ``j`` to ``j + 1`` in units of
    ``2 * pi / sectors[k - 1]``.

    :param sectors: Sector count of each ring, innermost first; each is a
        multiple of the one before it.
    :type sectors: tuple[int, ...]
    :param passages: Every open wall between two cells, as the unordered pair
        of ``(ring, sector)`` cells it joins; ``COURTYARD`` is the courtyard.
    :type passages: frozenset[frozenset[tuple[int, int]]]
    :param entrance: The outermost ring's sector whose outer wall is open.
    :type entrance: int

    """

    sectors: tuple[int, ...]
    passages: frozenset[frozenset[tuple[int, int]]]
    entrance: int

    def is_open(self, cell: tuple[int, int], other: tuple[int, int]) -> bool:
        """Return whether the wall between two neighboring cells is open.

        :param cell: One ``(ring, sector)`` cell.
        :type cell: tuple[int, int]
        :param other: The neighboring ``(ring, sector)`` cell.
        :type other: tuple[int, int]
        :returns: ``True`` if a passage joins the two cells.
        :rtype: bool

        """
        return frozenset((cell, other)) in self.passages


def ring_sectors(
    *, rings: int, inner_sectors: int, max_sectors: int
) -> tuple[int, ...]:
    """Choose each ring's sector count, doubling outward to keep cells square.

    :param rings: Rings of cells around the courtyard.
    :type rings: int
    :param inner_sectors: Sector count of the innermost ring.
    :type inner_sectors: int
    :param max_sectors: The most sectors any ring may have.
    :type max_sectors: int
    :returns: The sector count of each ring, innermost first.
    :rtype: tuple[int, ...]
    :raises TypeError: If any argument is not an int.
    :raises ValueError: If ``rings`` is not positive, ``inner_sectors`` is
        less than 3, or ``max_sectors`` is less than ``inner_sectors``.

    """
    IntValidator.type_is_int(value=rings)
    IntValidator.value_is_positive(value=rings)
    IntValidator.type_is_int(value=inner_sectors)
    IntValidator.type_is_int(value=max_sectors)
    # With two sectors, both radial walls of a ring would separate the same pair
    # of cells, so one passage would open both.
    if inner_sectors < _MIN_SECTORS:
        raise ValueError(
            f"inner_sectors must be at least {_MIN_SECTORS}, got {inner_sectors}"
        )
    if max_sectors < inner_sectors:
        raise ValueError(
            f"max_sectors must be at least inner_sectors={inner_sectors}, "
            f"got {max_sectors}"
        )

    sectors = [inner_sectors]
    for ring in range(2, rings + 1):
        doubled = 2 * sectors[-1]
        # Arc length of a doubled cell at the ring's middle radius, in ring depths.
        doubled_width = _TURN * (ring + 0.5) / doubled
        if doubled <= max_sectors and doubled_width >= _MIN_CELL_ASPECT:
            sectors.append(doubled)
        else:
            sectors.append(sectors[-1])
    return tuple(sectors)


def _neighbors(
    cell: tuple[int, int], sectors: tuple[int, ...]
) -> list[tuple[int, int]]:
    """List a ring cell's neighbors in its own ring and the adjacent ones.

    The courtyard is never listed: the search leaves it out and joins it to the
    maze separately, through a single doorway.

    :param cell: A ``(ring, sector)`` cell with ``ring >= 1``.
    :type cell: tuple[int, int]
    :param sectors: Sector count of each ring, innermost first.
    :type sectors: tuple[int, ...]
    :returns: The neighboring ring cells, sorted.
    :rtype: list[tuple[int, int]]

    """
    ring, sector = cell
    count = sectors[ring - 1]
    # The two neighbors in the same ring.
    candidates = {(ring, (sector - 1) % count), (ring, (sector + 1) % count)}
    # The one inward neighbor, whose sector spans this cell's own.
    if ring > 1:
        candidates.add((ring - 1, sector * sectors[ring - 2] // count))
    # Every outward neighbor within this cell's own angular span.
    if ring < len(sectors):
        ratio = sectors[ring] // count
        candidates.update((ring + 1, sector * ratio + i) for i in range(ratio))
    return sorted(candidates)


def carve_maze(
    *, rings: int, inner_sectors: int, max_sectors: int, seed: int
) -> CircularMaze:
    """Carve a perfect circular maze with an iterative randomized depth-first search.

    The search starts at the entrance -- the outermost ring's sector that
    starts at the top of the maze, in image coordinates -- and, at each step,
    moves to a random unvisited neighbor of the cell on top of its stack
    (opening the wall in between) or backtracks when there is none, so the
    passages form a spanning tree of the ring cells. One more passage then
    joins the courtyard to a random cell of the innermost ring, so the
    courtyard has exactly one way in and the maze has ``sum(sectors)``
    passages in total.

    :param rings: Rings of cells around the courtyard.
    :type rings: int
    :param inner_sectors: Sector count of the innermost ring.
    :type inner_sectors: int
    :param max_sectors: The most sectors any ring may have.
    :type max_sectors: int
    :param seed: Seed of ``numpy.random.default_rng``, the only source of
        randomness.
    :type seed: int
    :returns: The carved maze.
    :rtype: CircularMaze
    :raises TypeError: If any argument is not an int.
    :raises ValueError: If ``rings`` is not positive, ``inner_sectors`` is
        less than 3, ``max_sectors`` is less than ``inner_sectors``, or
        ``seed`` is negative.

    """
    sectors = ring_sectors(
        rings=rings, inner_sectors=inner_sectors, max_sectors=max_sectors
    )
    IntValidator.type_is_int(value=seed)
    IntValidator.value_is_non_negative(value=seed)

    rng = np.random.default_rng(seed=seed)
    # Angles run from +x towards +y, and +y points down in image coordinates,
    # so the top of the maze is three quarters of a turn.
    entrance = 3 * sectors[-1] // 4
    start = (rings, entrance)
    visited = {start}
    passages = set()
    stack = [start]
    while stack:
        cell = stack[-1]
        candidates = [c for c in _neighbors(cell, sectors) if c not in visited]
        if not candidates:
            # Dead end: backtrack to the previous cell on the path.
            stack.pop()
            continue
        chosen = candidates[rng.integers(low=0, high=len(candidates))]
        visited.add(chosen)
        passages.add(frozenset((cell, chosen)))
        stack.append(chosen)

    # The courtyard's single doorway into the innermost ring.
    doorway = (1, int(rng.integers(low=0, high=sectors[0])))
    passages.add(frozenset((COURTYARD, doorway)))
    return CircularMaze(
        sectors=sectors, passages=frozenset(passages), entrance=entrance
    )


# ========================================================================================
# Drawing
# ========================================================================================


def draw_maze(
    maze: CircularMaze,
    *,
    r: np.ndarray,
    theta: np.ndarray,
    maze_radius: float,
    wall_thickness: float,
) -> np.ndarray:
    """Evaluate a maze's image at arbitrary polar points.

    Every wall has the same physical thickness, ``wall_thickness`` ring
    depths: a point is on a circular wall if its radius is within half that of
    the wall's radius, and on a radial wall if its arc distance to the wall's
    angle is. An opening in a circular wall stops half a wall short of both
    ends of its arc, so wall corners stay closed.

    :param maze: The maze to draw.
    :type maze: CircularMaze
    :param r: Radii of the points, broadcastable against ``theta``.
    :type r: np.ndarray
    :param theta: Angles of the points, in radians, measured from ``+x``
        towards ``+y`` -- PyPFT's own angle convention.
    :type theta: np.ndarray
    :param maze_radius: Radius of the maze's outer wall.
    :type maze_radius: float
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth,
        in ``(0, 1)``.
    :type wall_thickness: float
    :returns: The image at every point, ``WALL_VALUE`` on a wall and
        ``CORRIDOR_VALUE`` everywhere else, with the broadcast shape of ``r``
        and ``theta``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``maze_radius`` is not positive or
        ``wall_thickness`` is not in ``(0, 1)``.

    """
    NumpyValidator.type_is_ndarray(value=r)
    NumpyValidator.type_is_ndarray(value=theta)
    FloatValidator.type_is_float(value=maze_radius)
    FloatValidator.value_is_positive(value=maze_radius)
    FloatValidator.type_is_float(value=wall_thickness)
    FloatValidator.value_is_positive(value=wall_thickness)
    if wall_thickness >= 1.0:
        raise ValueError(
            f"wall_thickness must be less than one ring's depth, got {wall_thickness}"
        )

    r, theta = np.broadcast_arrays(r, theta)
    rings = len(maze.sectors)
    depth = maze_radius / (rings + 1)
    half = wall_thickness * depth / 2
    wall = np.zeros(shape=r.shape, dtype=bool)

    # Circular walls: circle k (1 <= k <= rings + 1) separates ring k - 1 from
    # ring k, split into arcs along ring k's sectors; the outer wall, circle
    # rings + 1, follows the outermost ring's.
    for circle in range(1, rings + 2):
        count = maze.sectors[min(circle, rings) - 1]
        on_circle = np.abs(r - circle * depth) <= half
        # Which arc each point falls in, and its angle past that arc's start.
        position = np.mod(theta, _TURN) * count / _TURN
        arc = np.floor(position).astype(int) % count
        offset = (position - np.floor(position)) * _TURN / count
        # Only an arc's interior, half a wall clear of both ends, can be open.
        interior = (offset * r > half) & ((_TURN / count - offset) * r > half)
        if circle == rings + 1:
            open_arcs = np.arange(count) == maze.entrance
        else:
            inner_count = maze.sectors[circle - 2] if circle > 1 else 1
            open_arcs = np.array(
                [
                    maze.is_open(
                        (circle - 1, sector * inner_count // count), (circle, sector)
                    )
                    for sector in range(count)
                ]
            )
        wall |= on_circle & ~(interior & open_arcs[arc])

    # Radial walls: in ring k, the wall at the start of sector j separates it
    # from sector j - 1, spanning the ring's depth plus half a wall each way.
    for ring in range(1, rings + 1):
        count = maze.sectors[ring - 1]
        in_ring = (r >= ring * depth - half) & (r <= (ring + 1) * depth + half)
        for sector in range(count):
            if maze.is_open((ring, (sector - 1) % count), (ring, sector)):
                continue
            # Arc distance to the wall: the angle to it, wrapped to [-pi, pi),
            # times the radius.
            angle = sector * _TURN / count
            distance = np.abs(np.mod(theta - angle + np.pi, _TURN) - np.pi) * r
            wall |= in_ring & (distance <= half)

    return np.where(wall, WALL_VALUE, CORRIDOR_VALUE)


def _check_polar_sampling(
    maze: CircularMaze, *, n_angular: int, maze_fraction: float
) -> None:
    """Validate the arguments shared by both polar samplers.

    :param maze: The maze to sample.
    :type maze: CircularMaze
    :param n_angular: The number of spokes to sample on.
    :type n_angular: int
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of the
        sampled radius, in ``(0, 1]``.
    :type maze_fraction: float
    :raises TypeError: If ``maze_fraction`` is not a float.
    :raises ValueError: If a sector count of ``maze`` does not divide
        ``n_angular``, or ``maze_fraction`` is not in ``(0, 1]``.

    """
    FloatValidator.type_is_float(value=maze_fraction)
    FloatValidator.value_is_positive(value=maze_fraction)
    if maze_fraction > 1.0:
        raise ValueError(
            f"maze_fraction must be at most 1 (maze inside R), got {maze_fraction}"
        )
    if n_angular % maze.sectors[-1] != 0:
        raise ValueError(
            f"n_angular={n_angular} must be a multiple of the outermost ring's "
            f"{maze.sectors[-1]} sectors, so every radial wall lies on a sample angle"
        )


def _to_uint8(values: np.ndarray) -> np.ndarray:
    """Scale a ``[0, 1]`` maze image to ``[0, 255]`` and cast it to ``uint8``.

    :param values: The maze image, ``WALL_VALUE`` or ``CORRIDOR_VALUE`` everywhere.
    :type values: np.ndarray
    :returns: The 8-bit image.
    :rtype: np.ndarray

    """
    return np.round(values * _UINT8_MAX).astype(np.uint8)


def sample_maze_on_grid(
    maze: CircularMaze,
    *,
    grid: pypft.PolarGrid,
    maze_fraction: float,
    wall_thickness: float,
) -> np.ndarray:
    """Evaluate a maze at a ``PolarGrid``'s own sample points, as an 8-bit array.

    This is the exact path's sampling: every spoke at its own radii ``grid.r``,
    so the result feeds ``pypft.forward_pft`` directly.

    :param maze: The maze to sample.
    :type maze: CircularMaze
    :param grid: The grid to sample on; every sector count of ``maze`` must
        divide ``grid.n_angular``.
    :type grid: pypft.PolarGrid
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of
        ``grid.R``, in ``(0, 1]``.
    :type maze_fraction: float
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth.
    :type wall_thickness: float
    :returns: The ``(n_radial, n_angular)`` polar array, scaled to ``[0, 255]``
        and cast to ``uint8``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If a sector count of ``maze`` does not divide
        ``grid.n_angular``, or any other argument has an invalid value.

    """
    _check_polar_sampling(maze, n_angular=grid.n_angular, maze_fraction=maze_fraction)

    # grid.r is (n_angular, n_radial), one row per sample angle; the fixture is
    # stored in PyPFT's own (radial, angular) layout.
    values = draw_maze(
        maze,
        r=grid.r,
        theta=grid.theta[:, np.newaxis],
        maze_radius=maze_fraction * grid.R,
        wall_thickness=wall_thickness,
    ).T
    return _to_uint8(values=values)


def sample_maze_uniform_polar(
    maze: CircularMaze,
    *,
    n_radial: int,
    n_angular: int,
    radius: float,
    maze_fraction: float,
    wall_thickness: float,
) -> np.ndarray:
    """Evaluate a maze on a uniform polar grid, as an 8-bit array.

    The radii are ``pypft.cartesian_to_polar``'s: sample ``k`` at
    ``k * radius / n_radial``, ``k = 0 .. n_radial - 1``, the same on every
    spoke, and the spokes are centered and uniform, so the result is what
    uniform polar data of this maze looks like. It goes through
    ``pypft.resample_uniform_polar`` before ``pypft.forward_pft``.

    :param maze: The maze to sample.
    :type maze: CircularMaze
    :param n_radial: The number of uniform radial samples.
    :type n_radial: int
    :param n_angular: The number of spokes; every sector count of ``maze`` must
        divide it.
    :type n_angular: int
    :param radius: The radius the uniform samples cover.
    :type radius: float
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of
        ``radius``, in ``(0, 1]``.
    :type maze_fraction: float
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth.
    :type wall_thickness: float
    :returns: The ``(n_radial, n_angular)`` polar array, scaled to ``[0, 255]``
        and cast to ``uint8``.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If a sector count of ``maze`` does not divide
        ``n_angular``, or any other argument has an invalid value.

    """
    IntValidator.type_is_int(value=n_radial)
    IntValidator.value_is_positive(value=n_radial)
    IntValidator.type_is_int(value=n_angular)
    IntValidator.value_is_positive(value=n_angular)
    FloatValidator.type_is_float(value=radius)
    FloatValidator.value_is_positive(value=radius)
    _check_polar_sampling(maze, n_angular=n_angular, maze_fraction=maze_fraction)

    # Uniform radii down the rows and centered spoke angles across the columns:
    # PyPFT's (radial, angular) layout, as cartesian_to_polar returns it.
    radii = np.arange(n_radial) * (radius / n_radial)
    angles = harmonics(n_angular=n_angular) * (_TURN / n_angular)
    values = draw_maze(
        maze,
        r=radii[:, np.newaxis],
        theta=angles[np.newaxis, :],
        maze_radius=maze_fraction * radius,
        wall_thickness=wall_thickness,
    )
    return _to_uint8(values=values)


def sample_maze_rings(
    maze: CircularMaze,
    *,
    grid: pypft.PolarGrid,
    maze_fraction: float,
    wall_thickness: float,
    n_quadrature: int = DEFAULT_RING_QUADRATURE,
) -> np.ndarray:
    """Compute a maze's angular harmonics on each harmonic's own true rings.

    This is the ring-consistent route's input (``pypft.forward_pft_ring``'s first
    stage): harmonic ``n = grid.harmonics[i]`` from the maze evaluated
    analytically at ``n_quadrature`` equally spaced angles on rings at its own
    radii (row ``i`` of ``grid.r``), on ``pypft.dft.angular_dft``'s scale, so the
    result is the values of a ``pypft.PolarSpatialHarmonicSignal``. The maze is on
    ``sample_maze_on_grid``'s ``[0, 255]`` scale (walls ``0``), so the two routes'
    results compare directly. Nothing is written to disk: the exact path's
    ``maze_polar.tif`` stays the only maze fixture.

    :param maze: The maze to sample.
    :type maze: CircularMaze
    :param grid: The grid whose harmonics and rings to sample; every sector count
        of ``maze`` must divide ``grid.n_angular``.
    :type grid: pypft.PolarGrid
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of
        ``grid.R``, in ``(0, 1]``.
    :type maze_fraction: float
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth.
    :type wall_thickness: float
    :param n_quadrature: The number of angles evaluated around every ring.
    :type n_quadrature: int
    :returns: The ``(n_radial, n_angular)`` complex spatial-harmonic array.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If a sector count of ``maze`` does not divide
        ``grid.n_angular``, or any other argument has an invalid value.

    """
    _check_polar_sampling(maze, n_angular=grid.n_angular, maze_fraction=maze_fraction)
    IntValidator.type_is_int(value=n_quadrature)
    IntValidator.value_is_positive(value=n_quadrature)

    angles = np.arange(n_quadrature) * (_TURN / n_quadrature)
    result = np.empty((grid.n_radial, grid.n_angular), dtype=complex)
    radii = grid.r
    orders = np.abs(grid.harmonics)
    for order in np.unique(ar=orders):
        # Harmonics n and -n share their rings: evaluate the maze once per order,
        # at every quadrature angle, (radial, angle).
        rows = np.flatnonzero(a=orders == order)
        rings = _UINT8_MAX * draw_maze(
            maze,
            r=radii[rows[0]][:, np.newaxis],
            theta=angles[np.newaxis, :],
            maze_radius=maze_fraction * grid.R,
            wall_thickness=wall_thickness,
        )
        # The quadrature of exp(-i n phi), scaled to an n_angular-point DFT.
        phases = np.exp(-1j * np.outer(a=angles, b=grid.harmonics[rows]))
        result[:, rows] = rings @ phases * (grid.n_angular / n_quadrature)
    return result


def draw_maze_cartesian(
    maze: CircularMaze,
    *,
    size: int,
    radius: float,
    maze_fraction: float,
    wall_thickness: float,
) -> np.ndarray:
    """Evaluate a maze on a square pixel grid covering ``[-R, R]`` on both axes.

    Rows run downward, matching PyPFT's image convention: pixel ``(row, col)``
    sits at ``x`` growing along the columns and ``y`` growing down the rows,
    and its angle is measured from ``+x`` towards ``+y``.

    :param maze: The maze to draw.
    :type maze: CircularMaze
    :param size: Side length of the image, in pixels.
    :type size: int
    :param radius: Half the side length of the covered square, ``R``.
    :type radius: float
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of
        ``R``.
    :type maze_fraction: float
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth.
    :type wall_thickness: float
    :returns: The ``(size, size)`` 8-bit grayscale image.
    :rtype: np.ndarray
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If ``size`` is not positive, or any other argument has
        an invalid value.

    """
    IntValidator.type_is_int(value=size)
    IntValidator.value_is_positive(value=size)
    FloatValidator.type_is_float(value=radius)
    FloatValidator.value_is_positive(value=radius)

    # Pixel centers, then their polar coordinates.
    coordinates = np.linspace(start=-radius, stop=radius, num=size)
    x, y = np.meshgrid(coordinates, coordinates)
    values = draw_maze(
        maze,
        r=np.hypot(x, y),
        theta=np.arctan2(y, x),
        maze_radius=maze_fraction * radius,
        wall_thickness=wall_thickness,
    )
    return _to_uint8(values=values)


# ========================================================================================
# Output
# ========================================================================================


def make_maze(
    *,
    rings: int,
    inner_sectors: int,
    seed: int,
    wall_thickness: float,
    n_radial: int,
    n_angular: int,
    radius: float,
    maze_fraction: float,
    image_size: int,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Generate a maze on a ``PolarGrid`` and write all four files.

    The uniform polar file uses the grid's own ``n_radial``, ``n_angular`` and
    ``R``, so both polar files hold the same number of samples. Runs
    ``pypft.check_adequacy`` on the grid, which warns if ``n_radial`` is too
    small for ``n_angular``.

    :param rings: Rings of cells around the courtyard.
    :type rings: int
    :param inner_sectors: Sector count of the innermost ring; the outermost
        ring's count, ``inner_sectors`` doubled outward up to
        ``n_angular // SAMPLES_PER_SECTOR``, must divide ``n_angular``.
    :type inner_sectors: int
    :param seed: Seed of the maze's random number generator.
    :type seed: int
    :param wall_thickness: Wall thickness, as a fraction of one ring's depth.
    :type wall_thickness: float
    :param n_radial: The grid's radial sample count.
    :type n_radial: int
    :param n_angular: The grid's angular sample count.
    :type n_angular: int
    :param radius: The grid's space limit ``R``.
    :type radius: float
    :param maze_fraction: Radius of the maze's outer wall, as a fraction of
        ``R``.
    :type maze_fraction: float
    :param image_size: Side length, in pixels, of the two PNG files.
    :type image_size: int
    :param output_dir: Directory the four files are written to, created if
        missing.
    :type output_dir: Path
    :returns: The paths of ``POLAR_FILENAME``, ``UNIFORM_POLAR_FILENAME``,
        ``CARTESIAN_FILENAME``, and ``SOURCE_FILENAME`` under ``output_dir``, in
        that order.
    :rtype: tuple[Path, Path, Path, Path]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument has an invalid value.
    :raises PermissionError: If ``output_dir`` is not writable.

    """
    PathValidator.type_is_Path(value=output_dir)
    PathValidator.value_is_writable(value=output_dir)

    # The grid first, since it caps how many sectors a ring can have.
    grid = pypft.PolarGrid(n_radial=n_radial, n_angular=n_angular, R=radius)
    pypft.check_adequacy(grid=grid)

    # Carve the maze, then evaluate it on the polar grid, on a uniform polar grid
    # with the same sample counts, and on a pixel grid.
    maze = carve_maze(
        rings=rings,
        inner_sectors=inner_sectors,
        max_sectors=n_angular // SAMPLES_PER_SECTOR,
        seed=seed,
    )
    polar = sample_maze_on_grid(
        maze, grid=grid, maze_fraction=maze_fraction, wall_thickness=wall_thickness
    )
    uniform_polar = sample_maze_uniform_polar(
        maze,
        n_radial=n_radial,
        n_angular=n_angular,
        radius=radius,
        maze_fraction=maze_fraction,
        wall_thickness=wall_thickness,
    )
    source = draw_maze_cartesian(
        maze,
        size=image_size,
        radius=radius,
        maze_fraction=maze_fraction,
        wall_thickness=wall_thickness,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    polar_path = output_dir / POLAR_FILENAME
    uniform_polar_path = output_dir / UNIFORM_POLAR_FILENAME
    cartesian_path = output_dir / CARTESIAN_FILENAME
    source_path = output_dir / SOURCE_FILENAME

    # The fixture: an uncompressed 8-bit grayscale TIFF, (n_radial, n_angular).
    Image.fromarray(polar).save(polar_path)

    # The same maze on a uniform polar grid, in the same format.
    Image.fromarray(uniform_polar).save(uniform_polar_path)

    # The Cartesian display of the polar signal. Dropping the "Software" PNG
    # metadata keeps the file independent of the installed matplotlib version.
    signal = pypft.PolarSpatialAngularSignal(values=polar.astype(np.float64), grid=grid)
    figure = Figure(figsize=(_FIGURE_SIZE, _FIGURE_SIZE))
    pypft.render_cartesian(
        signal=signal, height=image_size, width=image_size, ax=figure.add_subplot()
    )
    figure.savefig(cartesian_path, metadata={"Software": None})

    # The maze evaluated directly on the pixel grid.
    Image.fromarray(source).save(source_path)
    return polar_path, uniform_polar_path, cartesian_path, source_path


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, generate the maze, and write the four files.

    :param argv: The command-line arguments, or ``None`` to read
        ``sys.argv[1:]``.
    :type argv: list[str] | None

    """
    parser = argparse.ArgumentParser(
        description="Generate a reproducible circular maze directly on a PolarGrid, "
        "for a test fixture."
    )
    parser.add_argument(
        "--rings",
        type=int,
        default=DEFAULT_RINGS,
        help=f"Rings of cells around the courtyard (default: {DEFAULT_RINGS}).",
    )
    parser.add_argument(
        "--inner-sectors",
        type=int,
        default=DEFAULT_INNER_SECTORS,
        help="Sector count of the innermost ring; outer rings double it as they "
        f"grow (default: {DEFAULT_INNER_SECTORS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed of the maze's random number generator (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=DEFAULT_WALL_THICKNESS,
        help="Wall thickness, as a fraction of one ring's depth "
        f"(default: {DEFAULT_WALL_THICKNESS}).",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=DEFAULT_N_RADIAL,
        help=f"The PolarGrid's radial sample count (default: {DEFAULT_N_RADIAL}).",
    )
    parser.add_argument(
        "--n-angular",
        type=int,
        default=DEFAULT_N_ANGULAR,
        help="The PolarGrid's angular sample count; rings have at most "
        f"n_angular // {SAMPLES_PER_SECTOR} sectors (default: {DEFAULT_N_ANGULAR}).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help=f"The PolarGrid's space limit R (default: {DEFAULT_RADIUS}).",
    )
    parser.add_argument(
        "--maze-fraction",
        type=float,
        default=DEFAULT_MAZE_FRACTION,
        help="Radius of the maze's outer wall, as a fraction of R "
        f"(default: {DEFAULT_MAZE_FRACTION}).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Side length, in pixels, of the two PNG files "
        f"(default: {DEFAULT_IMAGE_SIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the four files are written to (default: tests/samples/).",
    )
    args = parser.parse_args(args=argv)

    paths = make_maze(
        rings=args.rings,
        inner_sectors=args.inner_sectors,
        seed=args.seed,
        wall_thickness=args.wall_thickness,
        n_radial=args.n_radial,
        n_angular=args.n_angular,
        radius=args.radius,
        maze_fraction=args.maze_fraction,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
