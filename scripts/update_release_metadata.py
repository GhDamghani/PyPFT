from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pypft.release_metadata import apply_release_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update package version and changelog for a merged release PR."
        )
    )
    parser.add_argument(
        "--bump",
        choices=("major", "minor", "patch"),
        required=True,
        help="Release bump selected from the merged branch prefix.",
    )
    parser.add_argument(
        "--merge-body-file",
        required=True,
        help="Path to a UTF-8 file containing the merge commit body.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing pyproject.toml and CHANGELOG.md.",
    )
    parser.add_argument(
        "--release-date",
        default=None,
        help="Optional ISO date override. Defaults to today.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    merge_body_path = Path(args.merge_body_file).resolve()
    release_date = date.today()
    if args.release_date is not None:
        release_date = date.fromisoformat(args.release_date)

    update = apply_release_update(
        repo_root,
        bump_kind=args.bump,
        changelog_body=merge_body_path.read_text(encoding="utf-8"),
        release_date=release_date,
    )
    print(update.next_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
