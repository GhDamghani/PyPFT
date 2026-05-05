from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docs.yml"
DOCS_CONF = REPO_ROOT / "docs" / "source" / "conf.py"
DOCS_INDEX = REPO_ROOT / "docs" / "source" / "index.rst"
DOCS_REQUIREMENTS = REPO_ROOT / "requirements" / "docs.txt"


class TestDocsWorkflow(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow_text = DOCS_WORKFLOW.read_text(encoding="utf-8")

    def test_docs_workflow_triggers_on_pushes_to_main(self):
        self.assertIn("name: Docs", self.workflow_text)
        self.assertIn("push:\n    branches: [main]", self.workflow_text)

    def test_docs_workflow_builds_sphinx_html(self):
        self.assertIn("python-version: '3.14'", self.workflow_text)
        self.assertIn("python -m pip install -r requirements/docs.txt", self.workflow_text)
        self.assertIn("python -m pip install .", self.workflow_text)
        self.assertIn("python -m sphinx -W -b html docs/source docs/build/html", self.workflow_text)

    def test_docs_inputs_exist(self):
        self.assertTrue(DOCS_CONF.exists())
        self.assertTrue(DOCS_INDEX.exists())
        self.assertTrue(DOCS_REQUIREMENTS.exists())

    def test_docs_requirements_include_sphinx(self):
        self.assertIn("sphinx>=8.0", DOCS_REQUIREMENTS.read_text(encoding="utf-8"))