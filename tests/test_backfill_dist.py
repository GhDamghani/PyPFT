from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.backfill_dist import (
    EXPECTED_ARTIFACT_TYPES,
    build_distribution_name_pattern,
    collect_dist_artifacts,
    make_backfill_plan,
    parse_changelog_versions,
)


class TestBackfillDist(TestCase):
    def test_parse_changelog_versions_returns_semantic_versions(self):
        with TemporaryDirectory() as tmp_dir:
            changelog_path = Path(tmp_dir) / "CHANGELOG.md"
            changelog_path.write_text(
                "# Changelog\n\n## [v0.0.2] - 2026-05-01\n\n## [v0.0.10] - 2026-05-02\n",
                encoding="utf-8",
            )

            versions = parse_changelog_versions(changelog_path)

            self.assertEqual(versions, ["0.0.2", "0.0.10"])

    def test_collect_dist_artifacts_tracks_wheel_and_sdist_per_version(self):
        with TemporaryDirectory() as tmp_dir:
            dist_dir = Path(tmp_dir)
            (dist_dir / "pypft-0.0.1-py3-none-any.whl").write_text("", encoding="utf-8")
            (dist_dir / "pypft-0.0.1.tar.gz").write_text("", encoding="utf-8")
            (dist_dir / "pypft-0.0.2-py3-none-any.whl").write_text("", encoding="utf-8")

            artifacts = collect_dist_artifacts(dist_dir, "pypft")

            self.assertEqual(artifacts["0.0.1"], EXPECTED_ARTIFACT_TYPES)
            self.assertEqual(artifacts["0.0.2"], {"wheel"})

    def test_distribution_name_pattern_matches_normalized_names(self):
        pattern = build_distribution_name_pattern("py-pft")

        self.assertIsNotNone(pattern.match("py_pft-0.0.1-py3-none-any.whl"))
        self.assertIsNotNone(pattern.match("py-pft-0.0.1.tar.gz"))

    def test_make_backfill_plan_separates_buildable_and_unavailable_versions(self):
        plan = make_backfill_plan(
            current_version="0.0.3",
            changelog_versions=["0.0.1", "0.0.4"],
            tag_refs={"0.0.1": "v0.0.1", "0.0.2": "v0.0.2"},
            dist_artifacts={"0.0.1": EXPECTED_ARTIFACT_TYPES, "0.0.2": {"wheel"}},
        )

        self.assertEqual(plan.expected_versions, ["0.0.1", "0.0.2", "0.0.3", "0.0.4"])
        self.assertEqual(plan.missing_versions, ["0.0.2", "0.0.3", "0.0.4"])
        self.assertEqual(plan.buildable_versions, ["0.0.2", "0.0.3"])
        self.assertEqual(plan.unavailable_versions, ["0.0.4"])