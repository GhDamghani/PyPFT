from pathlib import Path
from unittest import TestCase

from scripts.sync_dependencies import (
    load_requirements,
    read_pyproject_dependencies,
    read_readme_dependencies,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestDependencySync(TestCase):
    def test_runtime_requirements_match_pyproject_dependencies(self):
        requirements = load_requirements(REPO_ROOT / "requirements" / "runtime.txt")
        pyproject_dependencies = read_pyproject_dependencies(REPO_ROOT / "pyproject.toml")
        self.assertEqual(requirements, pyproject_dependencies)

    def test_runtime_requirements_match_readme_dependencies(self):
        requirements = load_requirements(REPO_ROOT / "requirements" / "runtime.txt")
        readme_dependencies = read_readme_dependencies(REPO_ROOT / "README.md")
        self.assertEqual(requirements, readme_dependencies)