"""Tests for ``scripts/make_maze.py``, the generator of the committed maze fixture."""

import importlib.util
from collections import deque
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

import pypft

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "make_maze.py"
_SAMPLES_DIR = _REPO_ROOT / "tests" / "samples"


def _load_script() -> ModuleType:
    """Import ``scripts/make_maze.py`` as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name="make_maze", location=_SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module)
    return module


make_maze = _load_script()


def _carve(
    *, rings: int = 4, inner_sectors: int = 8, max_sectors: int = 32, seed: int = 0
):
    """Carve a maze with small, fast defaults."""
    return make_maze.carve_maze(
        rings=rings, inner_sectors=inner_sectors, max_sectors=max_sectors, seed=seed
    )


# ========================================================================================
# Ring layout
# ========================================================================================


def test_default_ring_layout() -> None:
    sectors = make_maze.ring_sectors(
        rings=make_maze.DEFAULT_RINGS,
        inner_sectors=make_maze.DEFAULT_INNER_SECTORS,
        max_sectors=make_maze.DEFAULT_N_ANGULAR // make_maze.SAMPLES_PER_SECTOR,
    )
    assert sectors == (8,) + (16,) * 7
    assert make_maze.DEFAULT_N_ANGULAR % sectors[-1] == 0


@pytest.mark.parametrize("rings", [1, 3, 8])
@pytest.mark.parametrize("max_sectors", [8, 24, 64])
def test_rings_sets_the_ring_count_and_sectors_double_outward(
    rings: int, max_sectors: int
) -> None:
    sectors = make_maze.ring_sectors(
        rings=rings, inner_sectors=8, max_sectors=max_sectors
    )
    assert len(sectors) == rings
    assert sectors[0] == 8
    for inner, outer in zip(sectors, sectors[1:]):
        assert outer in (inner, 2 * inner)
    assert max(sectors) <= max_sectors


# ========================================================================================
# Maze generation
# ========================================================================================


def test_same_seed_gives_identical_maze() -> None:
    assert _carve(seed=7) == _carve(seed=7)


def test_different_seed_gives_different_maze() -> None:
    assert _carve(seed=0).passages != _carve(seed=1).passages


@pytest.mark.parametrize("rings", [1, 2, 4, 7])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_maze_is_perfect(rings: int, seed: int) -> None:
    maze = _carve(rings=rings, seed=seed)
    cells = [make_maze.COURTYARD] + [
        (ring, sector)
        for ring, count in enumerate(maze.sectors, start=1)
        for sector in range(count)
    ]

    # A spanning tree: one passage fewer than cells, and every cell reachable.
    assert len(maze.passages) == len(cells) - 1
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for passage in maze.passages:
        a, b = sorted(passage)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    reached = {make_maze.COURTYARD}
    queue = deque([make_maze.COURTYARD])
    while queue:
        for neighbor in neighbors.get(queue.popleft(), []):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    assert reached == set(cells)

    # The courtyard has exactly one doorway, into the innermost ring.
    (doorway,) = neighbors[make_maze.COURTYARD]
    assert doorway[0] == 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rings": 0}, ValueError),
        ({"rings": 2.0}, TypeError),
        ({"inner_sectors": 2}, ValueError),
        ({"max_sectors": 4}, ValueError),
        ({"seed": -1}, ValueError),
    ],
)
def test_carve_maze_rejects_invalid_input(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        _carve(**kwargs)


# ========================================================================================
# Drawing
# ========================================================================================


def test_outer_wall_is_closed_except_at_the_entrance() -> None:
    maze = _carve()
    count = maze.sectors[-1]
    # One point in the middle of every outer-wall arc, exactly on the wall.
    theta = (np.arange(count) + 0.5) * 2 * np.pi / count
    values = make_maze.draw_maze(
        maze, r=np.ones(count), theta=theta, maze_radius=1.0, wall_thickness=0.25
    )
    expected = np.full(count, make_maze.WALL_VALUE)
    expected[maze.entrance] = make_maze.CORRIDOR_VALUE
    np.testing.assert_array_equal(values, expected)


def test_courtyard_center_and_outside_are_open() -> None:
    maze = _carve()
    values = make_maze.draw_maze(
        maze,
        r=np.array([0.0, 0.1, 1.2]),
        theta=np.zeros(3),
        maze_radius=1.0,
        wall_thickness=0.25,
    )
    np.testing.assert_array_equal(values, make_maze.CORRIDOR_VALUE)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_wall_between_neighbors_is_open_exactly_at_a_passage(seed: int) -> None:
    maze = _carve(seed=seed)
    rings = len(maze.sectors)
    depth = 1.0 / (rings + 1)
    r, theta, expected = [], [], []
    for ring, count in enumerate(maze.sectors, start=1):
        width = 2 * np.pi / count
        inner_count = maze.sectors[ring - 2] if ring > 1 else 1
        for sector in range(count):
            # The radial wall at the start of the sector, halfway through the ring.
            r.append((ring + 0.5) * depth)
            theta.append(sector * width)
            expected.append(maze.is_open((ring, (sector - 1) % count), (ring, sector)))
            # The circular wall on the sector's inner side, at the arc's middle.
            r.append(ring * depth)
            theta.append((sector + 0.5) * width)
            inner = (ring - 1, sector * inner_count // count)
            expected.append(maze.is_open(inner, (ring, sector)))
    values = make_maze.draw_maze(
        maze,
        r=np.array(r),
        theta=np.array(theta),
        maze_radius=1.0,
        wall_thickness=0.25,
    )
    np.testing.assert_array_equal(
        values,
        np.where(expected, make_maze.CORRIDOR_VALUE, make_maze.WALL_VALUE),
    )


def test_draw_maze_rejects_wall_thickness_of_a_full_ring() -> None:
    with pytest.raises(ValueError):
        make_maze.draw_maze(
            _carve(),
            r=np.ones(1),
            theta=np.zeros(1),
            maze_radius=1.0,
            wall_thickness=1.0,
        )


def test_sample_maze_rejects_n_angular_not_a_multiple_of_the_outer_sectors() -> None:
    maze = _carve(max_sectors=16)
    grid = pypft.PolarGrid(n_radial=8, n_angular=40, R=1.0)
    with pytest.raises(ValueError):
        make_maze.sample_maze(maze, grid=grid, maze_fraction=0.95, wall_thickness=0.25)


@pytest.mark.parametrize("maze_fraction", [0.0, 1.5])
def test_sample_maze_rejects_maze_fraction_outside_unit_interval(
    maze_fraction: float,
) -> None:
    grid = pypft.PolarGrid(n_radial=8, n_angular=48, R=1.0)
    with pytest.raises(ValueError):
        make_maze.sample_maze(
            _carve(max_sectors=16),
            grid=grid,
            maze_fraction=maze_fraction,
            wall_thickness=0.25,
        )


# ========================================================================================
# The committed fixture
# ========================================================================================


def test_defaults_reproduce_committed_fixture(tmp_path: Path) -> None:
    make_maze.main(argv=["--output-dir", str(tmp_path)])

    for filename in (make_maze.POLAR_FILENAME, make_maze.SOURCE_FILENAME):
        generated = np.asarray(Image.open(tmp_path / filename))
        committed = np.asarray(Image.open(_SAMPLES_DIR / filename))
        np.testing.assert_array_equal(generated, committed)

    polar = np.asarray(Image.open(tmp_path / make_maze.POLAR_FILENAME))
    assert polar.dtype == np.uint8
    assert polar.shape == (make_maze.DEFAULT_N_RADIAL, make_maze.DEFAULT_N_ANGULAR)
    assert (tmp_path / make_maze.CARTESIAN_FILENAME).is_file()
