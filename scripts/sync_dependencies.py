from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements" / "runtime.txt"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_README = REPO_ROOT / "README.md"
README_DEPENDENCY_START = "<!-- dependencies:start -->"
README_DEPENDENCY_END = "<!-- dependencies:end -->"


def load_requirements(requirements_path: Path) -> list[str]:
    dependencies: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", ".")):
            raise ValueError(
                "Only plain PEP 508 dependency lines are supported in the runtime requirements file."
            )
        dependencies.append(line)
    return dependencies


def read_pyproject_dependencies(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return list(data["project"]["dependencies"])


def render_dependency_block(dependencies: list[str]) -> str:
    lines = ["dependencies = ["]
    lines.extend(f'    "{dependency}",' for dependency in dependencies)
    lines.append("]")
    return "\n".join(lines)


def render_readme_dependency_block(dependencies: list[str]) -> str:
    lines = [README_DEPENDENCY_START, "```text"]
    lines.extend(dependencies)
    lines.append("```")
    lines.append(README_DEPENDENCY_END)
    return "\n".join(lines)


def sync_pyproject(pyproject_path: Path, requirements_path: Path) -> bool:
    dependencies = load_requirements(requirements_path)
    replacement = render_dependency_block(dependencies)
    content = pyproject_path.read_text(encoding="utf-8")

    marker = "dependencies = ["
    start = content.find(marker)
    if start == -1:
        raise ValueError(f"Could not find dependencies block in {pyproject_path}")

    end = content.find("]", start)
    if end == -1:
        raise ValueError(f"Could not find end of dependencies block in {pyproject_path}")

    updated_content = f"{content[:start]}{replacement}{content[end + 1:]}"
    if updated_content == content:
        return False

    pyproject_path.write_text(updated_content, encoding="utf-8")
    return True


def read_readme_dependencies(readme_path: Path) -> list[str]:
    content = readme_path.read_text(encoding="utf-8")
    start = content.find(README_DEPENDENCY_START)
    end = content.find(README_DEPENDENCY_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find dependency markers in {readme_path}")

    block = content[start:end].splitlines()
    dependencies: list[str] = []
    for line in block[2:]:
        stripped = line.strip()
        if stripped == "```":
            break
        if stripped:
            dependencies.append(stripped)
    return dependencies


def sync_readme(readme_path: Path, requirements_path: Path) -> bool:
    dependencies = load_requirements(requirements_path)
    replacement = render_readme_dependency_block(dependencies)
    content = readme_path.read_text(encoding="utf-8")

    start = content.find(README_DEPENDENCY_START)
    end = content.find(README_DEPENDENCY_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find dependency markers in {readme_path}")

    updated_content = (
        f"{content[:start]}{replacement}{content[end + len(README_DEPENDENCY_END):]}"
    )
    if updated_content == content:
        return False

    readme_path.write_text(updated_content, encoding="utf-8")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync project dependencies from requirements/runtime.txt into pyproject.toml."
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Path to the source requirements file.",
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
        "--check",
        action="store_true",
        help="Fail instead of rewriting when pyproject.toml is out of sync.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    requirements = load_requirements(args.requirements)
    pyproject_dependencies = read_pyproject_dependencies(args.pyproject)
    readme_dependencies = read_readme_dependencies(args.readme)
    if requirements == pyproject_dependencies and requirements == readme_dependencies:
        print("pyproject.toml and README.md dependencies are already in sync.")
        return 0

    if args.check:
        print("pyproject.toml or README.md dependencies are out of sync with the requirements file.")
        return 1

    sync_pyproject(args.pyproject, args.requirements)
    sync_readme(args.readme, args.requirements)
    print("Updated pyproject.toml and README.md dependencies from the requirements file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())