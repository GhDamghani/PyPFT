from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


RELEASE_BUMPS = {
    "major": "major",
    "minor": "minor",
    "patch": "patch",
    "hotfix": "patch",
}

CHANGELOG_HEADER = (
    "# Changelog\n\n"
    "All notable changes to this project will be documented in this file.\n\n"
)


def detect_release_class(branch_name: str) -> tuple[str, str]:
    prefix = branch_name.split("/", 1)[0].lower().strip()
    if prefix not in RELEASE_BUMPS:
        allowed = ", ".join(sorted(RELEASE_BUMPS))
        raise ValueError(
            f"Unsupported release branch prefix '{prefix}'. Expected one of: {allowed}."
        )
    return prefix, RELEASE_BUMPS[prefix]


def bump_version(current_version: str, bump_kind: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", current_version.strip())
    if match is None:
        raise ValueError(f"Unsupported semantic version: {current_version}")

    major, minor, patch = (int(part) for part in match.groups())

    if bump_kind == "major":
        return f"{major + 1}.0.0"
    if bump_kind == "minor":
        return f"{major}.{minor + 1}.0"
    if bump_kind == "patch":
        return f"{major}.{minor}.{patch + 1}"

    raise ValueError(f"Unsupported bump kind: {bump_kind}")


def read_current_version(pyproject_path: Path) -> str:
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find project version in {pyproject_path}")
    return match.group(1)


def update_pyproject_version(pyproject_path: Path, new_version: str) -> bool:
    content = pyproject_path.read_text(encoding="utf-8")
    updated_content, replacements = re.subn(
        r'(^version\s*=\s*")([^"]+)(")',
        rf'\g<1>{new_version}\g<3>',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError(f"Could not update project version in {pyproject_path}")
    if updated_content == content:
        return False
    pyproject_path.write_text(updated_content, encoding="utf-8")
    return True


def build_changelog_entry(
    new_version: str,
    release_class: str,
    pr_title: str,
    pr_number: str,
    release_date: str,
) -> str:
    release_heading = release_class.capitalize()
    return (
        f"## [v{new_version}] - {release_date}\n\n"
        f"### {release_heading}\n"
        f"- {pr_title} (#{pr_number})\n\n"
    )


def update_changelog(changelog_path: Path, entry: str, new_version: str) -> bool:
    heading = f"## [v{new_version}]"
    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
    else:
        existing = CHANGELOG_HEADER

    if heading in existing:
        return False

    if existing.startswith(CHANGELOG_HEADER):
        updated = f"{CHANGELOG_HEADER}{entry}{existing[len(CHANGELOG_HEADER):]}"
    else:
        updated = f"{CHANGELOG_HEADER}{entry}{existing}"

    changelog_path.write_text(updated, encoding="utf-8")
    return True


def write_github_output(output_path: str | None, values: dict[str, str]) -> None:
    if not output_path:
        return

    lines = [f"{key}={value}" for key, value in values.items()]
    with Path(output_path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_detect(args: argparse.Namespace) -> int:
    release_class, bump_kind = detect_release_class(args.branch)
    values = {
        "release_class": release_class,
        "bump_kind": bump_kind,
    }
    write_github_output(args.github_output, values)
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


def run_apply(args: argparse.Namespace) -> int:
    pyproject_path = Path(args.pyproject)
    changelog_path = Path(args.changelog)
    release_date = args.release_date or dt.date.today().isoformat()

    current_version = read_current_version(pyproject_path)
    bump_kind = RELEASE_BUMPS[args.release_class]
    new_version = bump_version(current_version, bump_kind)

    update_pyproject_version(pyproject_path, new_version)
    entry = build_changelog_entry(
        new_version=new_version,
        release_class=args.release_class,
        pr_title=args.pr_title,
        pr_number=str(args.pr_number),
        release_date=release_date,
    )
    update_changelog(changelog_path, entry, new_version)

    values = {
        "release_class": args.release_class,
        "bump_kind": bump_kind,
        "previous_version": current_version,
        "new_version": new_version,
    }
    write_github_output(args.github_output, values)
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update package version and changelog from release branch metadata."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect release class from branch name.")
    detect_parser.add_argument("--branch", required=True, help="Merged pull request source branch name.")
    detect_parser.add_argument(
        "--github-output",
        default=None,
        help="Optional path to the GitHub Actions output file.",
    )
    detect_parser.set_defaults(func=run_detect)

    apply_parser = subparsers.add_parser(
        "apply", help="Apply version and changelog updates for a release class."
    )
    apply_parser.add_argument(
        "--release-class",
        required=True,
        choices=sorted(RELEASE_BUMPS),
        help="Release class detected from the merged branch name.",
    )
    apply_parser.add_argument("--pyproject", required=True, help="Path to pyproject.toml.")
    apply_parser.add_argument("--changelog", required=True, help="Path to CHANGELOG.md.")
    apply_parser.add_argument("--pr-title", required=True, help="Merged pull request title.")
    apply_parser.add_argument("--pr-number", required=True, help="Merged pull request number.")
    apply_parser.add_argument(
        "--release-date",
        default=None,
        help="Optional release date in ISO format. Defaults to today.",
    )
    apply_parser.add_argument(
        "--github-output",
        default=None,
        help="Optional path to the GitHub Actions output file.",
    )
    apply_parser.set_defaults(func=run_apply)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())