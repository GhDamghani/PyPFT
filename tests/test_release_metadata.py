"""Tests for automated version bump and changelog update helpers.

This module covers the merge-driven release metadata utilities that update the
package version, insert changelog sections, and reject invalid or duplicate
release metadata edits.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pypft.release_metadata import (
    apply_release_update,
    compute_next_version,
    insert_release_entry,
    normalize_changelog_body,
)


def test_compute_next_version_uses_branch_bump_kind() -> None:
    assert compute_next_version("0.1.0.dev0", "patch") == "0.1.1"
    assert compute_next_version("0.1.0.dev0", "minor") == "0.2.0"
    assert compute_next_version("0.1.0.dev0", "major") == "1.0.0"


def test_normalize_changelog_body_uses_merge_description_fallback() -> None:
    assert normalize_changelog_body("\n\n") == (
        "- No changelog body was provided in the merge description."
    )


def test_insert_release_entry_rejects_duplicate_versions() -> None:
    changelog_text = (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [v0.1.0] - 2026-05-16\n\n"
        "### Minor\n\n"
        "- Existing entry\n"
    )

    with pytest.raises(ValueError):
        insert_release_entry(
            changelog_text,
            "## [v0.1.0] - 2026-05-16\n\n### Minor\n\n- Duplicate\n\n",
        )


def test_apply_release_update_updates_version_files_and_changelog(
    tmp_path: Path,
) -> None:
    _write_release_fixture(tmp_path)

    update = apply_release_update(
        tmp_path,
        bump_kind="minor",
        changelog_body="- Add workflow automation\n- Capture merge body",
        release_date=date(2026, 5, 16),
    )

    assert update.previous_version == "0.1.0.dev0"
    assert update.next_version == "0.2.0"

    pyproject_text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    init_text = (
        tmp_path / "src" / "pypft" / "__init__.py"
    ).read_text(encoding="utf-8")
    changelog_text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")

    assert 'version = "0.2.0"' in pyproject_text
    assert '__version__ = "0.2.0"' in init_text
    assert "## [v0.2.0] - 2026-05-16" in changelog_text
    assert "- Add workflow automation" in changelog_text
    assert changelog_text.index(
        "## [v0.2.0] - 2026-05-16"
    ) < changelog_text.index("## [v0.0.5] - 2026-04-28")


def _write_release_fixture(repo_root: Path) -> None:
    src_dir = repo_root / "src" / "pypft"
    src_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "PyPFT"\n'
        'version = "0.1.0.dev0"\n',
        encoding="utf-8",
    )
    (src_dir / "__init__.py").write_text(
        '__version__ = "0.1.0.dev0"\n',
        encoding="utf-8",
    )
    (repo_root / "CHANGELOG.md").write_text(
        (
            "# Changelog\n\n"
            "All notable changes to this project will be documented in "
            "this file.\n\n"
            "## [Unreleased]\n\n"
            "- Pending unreleased note\n\n"
            "## [v0.0.5] - 2026-04-28\n\n"
            "### Patch\n\n"
            "- Fix unprotect_reviews (#4)\n"
        ),
        encoding="utf-8",
    )
