from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


EXPECTED_ARTIFACT_TYPES = frozenset({"sdist", "wheel"})
RELEASE_VERSION_PATTERN = re.compile(r"^## \[v(\d+\.\d+\.\d+)\]", re.MULTILINE)
TAG_VERSION_PATTERN = re.compile(r"^v(\d+\.\d+\.\d+)$")


@dataclass(frozen=True)
class BackfillPlan:
    expected_versions: list[str]
    missing_versions: list[str]
    buildable_versions: list[str]
    unavailable_versions: list[str]


def version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def build_distribution_name_pattern(project_name: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in re.split(r"[-_.]+", project_name) if piece]
    joined = r"[-_.]+".join(pieces)
    return re.compile(rf"^{joined}-(\d+\.\d+\.\d+)(?:\.tar\.gz|-.+\.whl)$", re.IGNORECASE)


def read_project_metadata(pyproject_path: Path) -> tuple[str, str]:
    with pyproject_path.open("rb") as handle:
        content = tomllib.load(handle)

    project = content.get("project", {})
    name = project.get("name")
    version = project.get("version")
    if not name or not version:
        raise ValueError(f"Could not read project.name and project.version from {pyproject_path}")
    return str(name), str(version)


def parse_changelog_versions(changelog_path: Path) -> list[str]:
    if not changelog_path.exists():
        return []
    content = changelog_path.read_text(encoding="utf-8")
    return sorted({match.group(1) for match in RELEASE_VERSION_PATTERN.finditer(content)}, key=version_key)


def collect_dist_artifacts(dist_dir: Path, project_name: str) -> dict[str, set[str]]:
    artifacts: dict[str, set[str]] = {}
    if not dist_dir.exists():
        return artifacts

    filename_pattern = build_distribution_name_pattern(project_name)
    for child in dist_dir.iterdir():
        if not child.is_file():
            continue

        match = filename_pattern.match(child.name)
        if match is None:
            continue

        version = match.group(1)
        artifact_type = "wheel" if child.name.endswith(".whl") else "sdist"
        artifacts.setdefault(version, set()).add(artifact_type)

    return artifacts


def run_git(args: list[str], repo_root: Path, *, capture_output: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return result.stdout.strip() if capture_output else ""


def collect_tag_refs(repo_root: Path, *, fetch_tags: bool = False) -> dict[str, str]:
    if fetch_tags:
        run_git(["fetch", "--tags"], repo_root, capture_output=False)

    output = run_git(["tag"], repo_root)
    refs: dict[str, str] = {}
    for line in output.splitlines():
        tag = line.strip()
        match = TAG_VERSION_PATTERN.fullmatch(tag)
        if match is None:
            continue
        refs[match.group(1)] = tag
    return refs


def make_backfill_plan(
    *,
    current_version: str,
    changelog_versions: list[str],
    tag_refs: dict[str, str],
    dist_artifacts: dict[str, set[str]],
) -> BackfillPlan:
    expected_versions = sorted(
        set(changelog_versions) | set(tag_refs) | {current_version},
        key=version_key,
    )
    missing_versions = [
        version
        for version in expected_versions
        if dist_artifacts.get(version, set()) != EXPECTED_ARTIFACT_TYPES
    ]
    buildable_versions = [
        version for version in missing_versions if version in tag_refs or version == current_version
    ]
    unavailable_versions = [version for version in missing_versions if version not in buildable_versions]
    return BackfillPlan(
        expected_versions=expected_versions,
        missing_versions=missing_versions,
        buildable_versions=buildable_versions,
        unavailable_versions=unavailable_versions,
    )


def build_source_tree(source_dir: Path, dist_dir: Path, python_executable: str) -> None:
    subprocess.run(
        [python_executable, "-m", "build", "--outdir", str(dist_dir), str(source_dir)],
        check=True,
    )


def build_version_from_ref(repo_root: Path, ref: str, dist_dir: Path, python_executable: str) -> None:
    with tempfile.TemporaryDirectory(prefix="pypft-dist-") as temp_dir:
        worktree_dir = Path(temp_dir) / "source"
        run_git(["worktree", "add", "--detach", str(worktree_dir), ref], repo_root, capture_output=False)
        try:
            build_source_tree(worktree_dir, dist_dir, python_executable)
        finally:
            run_git(["worktree", "remove", "--force", str(worktree_dir)], repo_root, capture_output=False)


def backfill_dist(
    repo_root: Path,
    *,
    pyproject_path: Path,
    changelog_path: Path,
    dist_dir: Path,
    python_executable: str,
    fetch_tags: bool = False,
    dry_run: bool = False,
) -> BackfillPlan:
    project_name, current_version = read_project_metadata(pyproject_path)
    changelog_versions = parse_changelog_versions(changelog_path)
    tag_refs = collect_tag_refs(repo_root, fetch_tags=fetch_tags)
    dist_artifacts = collect_dist_artifacts(dist_dir, project_name)
    plan = make_backfill_plan(
        current_version=current_version,
        changelog_versions=changelog_versions,
        tag_refs=tag_refs,
        dist_artifacts=dist_artifacts,
    )

    print(f"Known versions: {', '.join(plan.expected_versions) or 'none'}")
    print(f"Missing from dist: {', '.join(plan.missing_versions) or 'none'}")

    if plan.unavailable_versions:
        unavailable = ", ".join(plan.unavailable_versions)
        raise RuntimeError(
            "Cannot build all missing versions because no matching git ref was found for: "
            f"{unavailable}. Fetch tags or create matching v<version> tags first."
        )

    if dry_run or not plan.buildable_versions:
        return plan

    dist_dir.mkdir(parents=True, exist_ok=True)
    for version in plan.buildable_versions:
        ref = tag_refs.get(version)
        if ref is None:
            print(f"Building current working tree for {version}")
            build_source_tree(repo_root, dist_dir, python_executable)
            continue

        print(f"Building {version} from {ref}")
        build_version_from_ref(repo_root, ref, dist_dir, python_executable)

    return plan


def build_parser() -> argparse.ArgumentParser:
    default_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build missing package artifacts into dist for known project versions."
    )
    parser.add_argument(
        "--repo-root",
        default=str(default_repo_root),
        help="Repository root. Defaults to the project root containing this script.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml relative to --repo-root.",
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md relative to --repo-root.",
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Distribution output directory relative to --repo-root.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run python -m build.",
    )
    parser.add_argument(
        "--fetch-tags",
        action="store_true",
        help="Run git fetch --tags before resolving version refs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which versions are missing without building them.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    try:
        backfill_dist(
            repo_root,
            pyproject_path=repo_root / args.pyproject,
            changelog_path=repo_root / args.changelog,
            dist_dir=repo_root / args.dist_dir,
            python_executable=args.python,
            fetch_tags=args.fetch_tags,
            dry_run=args.dry_run,
        )
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())