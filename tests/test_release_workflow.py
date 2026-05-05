from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


class TestReleaseWorkflow(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow_text = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    def test_release_workflow_triggers_on_closed_pull_requests_to_main(self):
        self.assertIn("name: Release", self.workflow_text)
        self.assertIn("pull_request:\n    types: [closed]\n    branches: [main]", self.workflow_text)

    def test_release_job_only_runs_for_merged_pull_requests(self):
        self.assertIn("if: github.event.pull_request.merged == true", self.workflow_text)
        self.assertIn("uses: python-semantic-release/python-semantic-release@v9.20.0", self.workflow_text)

    def test_branch_prefix_classification_matches_supported_release_classes(self):
        self.assertIn('if [[ "$BRANCH" == major/* ]]; then', self.workflow_text)
        self.assertIn('elif [[ "$BRANCH" == minor/* ]]; then', self.workflow_text)
        self.assertIn('elif [[ "$BRANCH" == patch/* ]]; then', self.workflow_text)
        self.assertNotIn('hotfix/*', self.workflow_text)
        self.assertIn('echo "Unsupported branch prefix: $BRANCH"', self.workflow_text)
        self.assertIn("exit 1", self.workflow_text)