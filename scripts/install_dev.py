from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import venv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local development environment."
    )
    parser.add_argument(
        "--venv-dir",
        default=".venv",
        help="Directory used for the virtual environment.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Install the optional CuPy GPU extra for CUDA 12.x.",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info[:2] != (3, 14):
        raise SystemExit(
            "PyPFT currently targets Python 3.14.x for development."
        )

    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    venv_dir = root / args.venv_dir
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)

    if sys.platform == "win32":
        python_executable = venv_dir / "Scripts" / "python.exe"
    else:
        python_executable = venv_dir / "bin" / "python"

    extras = ".[dev,docs,bench]"
    if args.gpu:
        extras = ".[dev,docs,bench,gpu]"

    subprocess.run(
        [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        cwd=root,
    )
    subprocess.run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "-e",
            extras,
        ],
        check=True,
        cwd=root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
