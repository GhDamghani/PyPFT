import tomllib
from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DEV_REQUIREMENTS = REPO_ROOT / "requirements" / "dev.txt"
TASKS_FILE = REPO_ROOT / "tasks.py"
README = REPO_ROOT / "README.md"


class TestInvokeTasks(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks_text = TASKS_FILE.read_text(encoding="utf-8")
        cls.readme_text = README.read_text(encoding="utf-8")
        cls.dev_requirements_text = DEV_REQUIREMENTS.read_text(encoding="utf-8")
        with PYPROJECT.open("rb") as handle:
            cls.pyproject = tomllib.load(handle)

    def test_dev_extra_includes_invoke_and_docs_tools(self):
        dev_dependencies = self.pyproject["project"]["optional-dependencies"]["dev"]

        self.assertIn("invoke>=2.2", dev_dependencies)
        self.assertIn("sphinx>=8.0", dev_dependencies)
        self.assertIn("build>=1.2", dev_dependencies)

    def test_tasks_file_exposes_expected_commands(self):
        self.assertIn("@task", self.tasks_text)
        self.assertIn('@task(name="sync-deps")', self.tasks_text)
        self.assertIn('@task(name="backfill-dist")', self.tasks_text)
        self.assertIn("def install(", self.tasks_text)
        self.assertIn("def sync_deps(", self.tasks_text)
        self.assertIn("def backfill_dist(", self.tasks_text)
        self.assertIn("def docs(", self.tasks_text)
        self.assertIn('args.append("--dev")', self.tasks_text)
        self.assertIn('"install_env.py"', self.tasks_text)
        self.assertIn('"sync_dependencies.py"', self.tasks_text)
        self.assertIn('"backfill_dist.py"', self.tasks_text)
        self.assertIn('[PYTHON, "-m", "sphinx"]', self.tasks_text)

    def test_readme_mentions_invoke_usage(self):
        self.assertIn("scripts/install_env.py --dev", self.readme_text)
        self.assertIn("inv --list", self.readme_text)
        self.assertIn("inv docs", self.readme_text)

    def test_dev_requirements_include_invoke(self):
        self.assertIn("invoke>=2.2", self.dev_requirements_text)