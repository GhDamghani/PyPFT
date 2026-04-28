from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.update_release_metadata import (
    CHANGELOG_HEADER,
    build_changelog_entry,
    bump_version,
    detect_release_class,
    read_current_version,
    update_changelog,
    update_pyproject_version,
)


class TestReleaseUpdate(TestCase):
    def test_detect_release_class(self):
        self.assertEqual(detect_release_class("major/add-breaking-change"), ("major", "major"))
        self.assertEqual(detect_release_class("minor/new-feature"), ("minor", "minor"))
        self.assertEqual(detect_release_class("patch/fix-docs"), ("patch", "patch"))
        self.assertEqual(detect_release_class("hotfix/outage-fix"), ("hotfix", "patch"))

    def test_detect_release_class_rejects_unknown_prefix(self):
        with self.assertRaises(ValueError):
            detect_release_class("feature/new-work")

    def test_bump_version(self):
        self.assertEqual(bump_version("0.0.4", "major"), "1.0.0")
        self.assertEqual(bump_version("0.0.4", "minor"), "0.1.0")
        self.assertEqual(bump_version("0.0.4", "patch"), "0.0.5")

    def test_update_pyproject_version(self):
        with TemporaryDirectory() as tmp_dir:
            pyproject_path = Path(tmp_dir) / "pyproject.toml"
            pyproject_path.write_text(
                "[project]\nname = \"pypft\"\nversion = \"0.0.4\"\n",
                encoding="utf-8",
            )

            changed = update_pyproject_version(pyproject_path, "0.1.0")

            self.assertTrue(changed)
            self.assertEqual(read_current_version(pyproject_path), "0.1.0")

    def test_update_changelog_creates_and_prepends_entries(self):
        with TemporaryDirectory() as tmp_dir:
            changelog_path = Path(tmp_dir) / "CHANGELOG.md"
            first_entry = build_changelog_entry(
                new_version="0.0.5",
                release_class="hotfix",
                pr_title="Fix cache invalidation",
                pr_number="12",
                release_date="2026-04-28",
            )
            second_entry = build_changelog_entry(
                new_version="0.1.0",
                release_class="minor",
                pr_title="Add release automation",
                pr_number="15",
                release_date="2026-04-29",
            )

            created = update_changelog(changelog_path, first_entry, "0.0.5")
            prepended = update_changelog(changelog_path, second_entry, "0.1.0")

            content = changelog_path.read_text(encoding="utf-8")
            self.assertTrue(created)
            self.assertTrue(prepended)
            self.assertTrue(content.startswith(CHANGELOG_HEADER))
            self.assertLess(content.index("## [v0.1.0]"), content.index("## [v0.0.5]"))

    def test_update_changelog_is_idempotent_for_same_version(self):
        with TemporaryDirectory() as tmp_dir:
            changelog_path = Path(tmp_dir) / "CHANGELOG.md"
            entry = build_changelog_entry(
                new_version="0.0.5",
                release_class="patch",
                pr_title="Tighten tests",
                pr_number="21",
                release_date="2026-04-28",
            )

            first = update_changelog(changelog_path, entry, "0.0.5")
            second = update_changelog(changelog_path, entry, "0.0.5")

            self.assertTrue(first)
            self.assertFalse(second)