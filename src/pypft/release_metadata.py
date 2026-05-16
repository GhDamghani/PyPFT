from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Literal
import tomllib

BumpKind = Literal["major", "minor", "patch"]

_SEMVER_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
)
_PYPROJECT_VERSION_PATTERN = re.compile(
    r'(?m)^version = "(?P<version>[^"]+)"$'
)
_INIT_VERSION_PATTERN = re.compile(
    r'(?m)^__version__ = "(?P<version>[^"]+)"$'
)
_RELEASE_HEADING_PATTERN = re.compile(
    r"^## \[v[^\]]+\] - \d{4}-\d{2}-\d{2}$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ReleaseUpdate:
    previous_version: str
    next_version: str
    bump_kind: BumpKind
    release_date: date
    changelog_body: str


def read_project_version(pyproject_path: Path) -> str:
    with pyproject_path.open("rb") as handle:
        project_table = tomllib.load(handle)["project"]
    version = project_table["version"]
    if not isinstance(version, str):
        raise ValueError("Expected [project].version to be a string.")
    return version


def compute_next_version(current_version: str, bump_kind: BumpKind) -> str:
    match = _SEMVER_PATTERN.match(current_version)
    if match is None:
        raise ValueError(
            f"Unsupported version format {current_version!r}; expected semver."
        )

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    if bump_kind == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_kind == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    return f"{major}.{minor}.{patch}"


def normalize_changelog_body(body: str) -> str:
    stripped = body.strip()
    if stripped:
        return stripped
    return "- No changelog body was provided in the merge description."


def format_release_entry(update: ReleaseUpdate) -> str:
    section_title = update.bump_kind.capitalize()
    return (
        f"## [v{update.next_version}] - {update.release_date.isoformat()}\n\n"
        f"### {section_title}\n\n"
        f"{normalize_changelog_body(update.changelog_body)}\n\n"
    )


def replace_pyproject_version(pyproject_text: str, new_version: str) -> str:
    updated_text, replacements = _PYPROJECT_VERSION_PATTERN.subn(
        f'version = "{new_version}"',
        pyproject_text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(
            "Could not update the project version in pyproject.toml."
        )
    return updated_text


def replace_init_version(init_text: str, new_version: str) -> str:
    updated_text, replacements = _INIT_VERSION_PATTERN.subn(
        f'__version__ = "{new_version}"',
        init_text,
        count=1,
    )
    if replacements != 1:
        raise ValueError(
            "Could not update __version__ in src/pypft/__init__.py."
        )
    return updated_text


def insert_release_entry(changelog_text: str, release_entry: str) -> str:
    version_heading = release_entry.splitlines()[0]
    if version_heading in changelog_text:
        raise ValueError(
            f"Changelog already contains a release entry for "
            f"{version_heading}."
        )

    first_release_match = _RELEASE_HEADING_PATTERN.search(changelog_text)
    if first_release_match is None:
        changelog = changelog_text.rstrip()
        return f"{changelog}\n\n{release_entry}"

    insertion_point = first_release_match.start()
    prefix = changelog_text[:insertion_point].rstrip()
    suffix = changelog_text[insertion_point:].lstrip()
    return f"{prefix}\n\n{release_entry}{suffix}"


def apply_release_update(
    repo_root: Path,
    *,
    bump_kind: BumpKind,
    changelog_body: str,
    release_date: date,
) -> ReleaseUpdate:
    pyproject_path = repo_root / "pyproject.toml"
    init_path = repo_root / "src" / "pypft" / "__init__.py"
    changelog_path = repo_root / "CHANGELOG.md"

    previous_version = read_project_version(pyproject_path)
    next_version = compute_next_version(previous_version, bump_kind)
    update = ReleaseUpdate(
        previous_version=previous_version,
        next_version=next_version,
        bump_kind=bump_kind,
        release_date=release_date,
        changelog_body=changelog_body,
    )

    pyproject_path.write_text(
        replace_pyproject_version(
            pyproject_path.read_text(encoding="utf-8"),
            next_version,
        ),
        encoding="utf-8",
    )
    init_path.write_text(
        replace_init_version(
            init_path.read_text(encoding="utf-8"),
            next_version,
        ),
        encoding="utf-8",
    )
    changelog_path.write_text(
        insert_release_entry(
            changelog_path.read_text(encoding="utf-8"),
            format_release_entry(update),
        ),
        encoding="utf-8",
    )
    return update


__all__ = [
    "BumpKind",
    "ReleaseUpdate",
    "apply_release_update",
    "compute_next_version",
    "format_release_entry",
    "insert_release_entry",
    "normalize_changelog_body",
    "read_project_version",
    "replace_init_version",
    "replace_pyproject_version",
]
