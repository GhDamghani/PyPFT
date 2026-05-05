from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

from sync_dependencies import DEFAULT_PYPROJECT, DEFAULT_README, DEFAULT_REQUIREMENTS, sync_pyproject, sync_readme


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV = REPO_ROOT / ".venv"
REQUIRED_PYTHON = (3, 14)


def current_python_is_supported() -> bool:
    return sys.version_info[:2] == REQUIRED_PYTHON


def installation_guidance() -> str:
    version_label = f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"
    system = platform.system()
    if system == "Windows":
        return textwrap.dedent(
            f"""
            Python {version_label} is required.

            Install it with one of these options:
            - winget install Python.Python.{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}
            - Download the installer from https://www.python.org/downloads/

            Then rerun this script with:
            - py -{version_label} scripts/install_env.py
            """
        ).strip()
    if system == "Darwin":
        return textwrap.dedent(
            f"""
            Python {version_label} is required.

            Install it with one of these options:
            - brew install python@{version_label}
            - Download the installer from https://www.python.org/downloads/

            Then rerun this script with:
            - python{version_label} scripts/install_env.py
            """
        ).strip()
    return textwrap.dedent(
        f"""
        Python {version_label} is required.

        Install it with your package manager, pyenv, or from https://www.python.org/downloads/.
        Then rerun this script with:
        - python{version_label} scripts/install_env.py
        """
    ).strip()


def venv_python_path(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def run_command(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Python 3.14 virtual environment and install PyPFT dependencies."
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV,
        help="Path where the virtual environment should be created.",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Path to the runtime dependency file.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_README,
        help="Path to README.md.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without creating a virtual environment or installing anything.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not current_python_is_supported():
        print(installation_guidance(), file=sys.stderr)
        return 1

    sync_pyproject(args.pyproject, args.requirements)
    sync_readme(args.readme, args.requirements)

    venv_path = args.venv_path.resolve()
    if args.dry_run:
        print(f"Would create virtual environment at {venv_path}")
    else:
        venv.EnvBuilder(with_pip=True).create(venv_path)
        print(f"Created virtual environment at {venv_path}")

    venv_python = venv_python_path(venv_path)
    run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)
    run_command([str(venv_python), "-m", "pip", "install", "-r", str(args.requirements)], args.dry_run)
    run_command(
        [str(venv_python), "-m", "pip", "install", "-e", str(REPO_ROOT), "--no-deps"],
        args.dry_run,
    )

    if platform.system() == "Windows":
        activation_command = f"{venv_path}\\Scripts\\Activate.ps1"
    else:
        activation_command = f"source {venv_path}/bin/activate"

    print(f"Activate the environment with: {activation_command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())