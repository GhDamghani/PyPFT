from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "update-release-metadata.yml"


class TestUpdateReleaseMetadataWorkflow(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow_text = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_triggers_on_closed_pull_requests(self):
        self.assertIn("name: Update release metadata", self.workflow_text)
        self.assertIn("pull_request:\n    types:\n      - closed", self.workflow_text)
        self.assertIn("github.event.pull_request.merged == true && github.event.pull_request.base.ref == 'main'", self.workflow_text)

    def test_workflow_passes_pr_body_to_release_script(self):
        self.assertIn("pull-requests: read", self.workflow_text)
        self.assertIn("PR_BODY: ${{ github.event.pull_request.body }}", self.workflow_text)
        self.assertIn('--pr-body "$PR_BODY"', self.workflow_text)
        self.assertIn("python scripts/update_release_metadata.py apply", self.workflow_text)

    def test_workflow_builds_and_commits_local_distribution(self):
        self.assertIn("python -m pip install --upgrade build", self.workflow_text)
        self.assertIn("python -m build --outdir dist", self.workflow_text)
        self.assertNotIn("rm -rf dist", self.workflow_text)
        self.assertIn('git status --porcelain -- pyproject.toml CHANGELOG.md dist', self.workflow_text)
        self.assertIn("git add -f dist", self.workflow_text)
        self.assertIn("CasperWA/push-protected@v2", self.workflow_text)
        self.assertIn("- Local distribution: dist/", self.workflow_text)
        self.assertIn("## Release metadata updated", self.workflow_text)