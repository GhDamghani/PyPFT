#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


PYPROJECT = Path("pyproject.toml")
CHANGELOG = Path("CHANGELOG.md")


VERSION_RE = re.compile(r'(?m)^version\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$')


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "SemVer":
        m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value.strip())
        if not m:
            raise ValueError(f"Invalid semantic version: {value}")
        return cls(*(int(part) for part in m.groups()))

    def bump(self, level: str) -> "SemVer":
        match level:
            case "major":
                return SemVer(self.major + 1, 0, 0)
            case "minor":
                return SemVer(self.major, self.minor + 1, 0)
            case "patch":
                return SemVer(self.major, self.minor, self.patch + 1)
            case _:
                raise ValueError(f"Unsupported bump level: {level}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def read_current_version() -> SemVer:
    text = PYPROJECT.read_text(encoding="utf-8")
    m = VERSION_RE.search(text)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return SemVer.parse(m.group("version"))


def write_version(new_version: SemVer) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    updated, count = VERSION_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise RuntimeError("Failed to update version in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8")


def render_changelog_entry(
    version: SemVer,
    pr_number: str,
    pr_title: str,
    branch: str,
    notes: str,
) -> str:
    date = datetime.now(UTC).date().isoformat()
    notes = notes.strip() or "- No additional changelog notes provided."
    return (
        f"## [{version}] - {date}\n\n"
        f"PR: #{pr_number} — {pr_title}\n\n"
        f"Source branch: `{branch}`\n\n"
        f"{notes}\n\n"
    )


def update_changelog(entry: str) -> None:
    if CHANGELOG.exists():
        existing = CHANGELOG.read_text(encoding="utf-8")
    else:
        existing = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"

    if existing.startswith("# Changelog"):
        head, _, tail = existing.partition("\n\n")
        new_text = f"{head}\n\n{entry}{tail}"
    else:
        new_text = f"# Changelog\n\n{entry}{existing}"

    CHANGELOG.write_text(new_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["major", "minor", "patch"])
    parser.add_argument("--pr-number")
    parser.add_argument("--pr-title")
    parser.add_argument("--branch")
    parser.add_argument("--notes-file")
    parser.add_argument("--print-version", action="store_true")
    args = parser.parse_args()

    if args.print_version:
        print(read_current_version())
        return

    assert args.level
    assert args.pr_number
    assert args.pr_title
    assert args.branch
    assert args.notes_file

    current = read_current_version()
    new_version = current.bump(args.level)
    write_version(new_version)

    notes = Path(args.notes_file).read_text(encoding="utf-8")
    entry = render_changelog_entry(
        version=new_version,
        pr_number=args.pr_number,
        pr_title=args.pr_title,
        branch=args.branch,
        notes=notes,
    )
    update_changelog(entry)


if __name__ == "__main__":
    main()
