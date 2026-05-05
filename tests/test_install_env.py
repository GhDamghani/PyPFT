from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts import install_env


class TestInstallEnv(TestCase):
    def test_build_parser_supports_dev_install(self):
        parser = install_env.build_parser()

        args = parser.parse_args(["--dev"])

        self.assertTrue(args.dev)
        self.assertEqual(args.dev_requirements, install_env.DEFAULT_DEV_REQUIREMENTS)

    def test_main_upgrades_pip_before_runtime_and_dev_installs(self):
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            runtime_requirements = repo_root / "runtime.txt"
            dev_requirements = repo_root / "dev.txt"
            pyproject = repo_root / "pyproject.toml"
            readme = repo_root / "README.md"
            venv_path = repo_root / ".venv"
            venv_python = venv_path / "Scripts" / "python.exe"

            for path in (runtime_requirements, dev_requirements, pyproject, readme):
                path.write_text("", encoding="utf-8")

            commands: list[list[str]] = []

            with (
                patch.object(install_env.sys, "argv", [
                    "install_env.py",
                    "--venv-path",
                    str(venv_path),
                    "--requirements",
                    str(runtime_requirements),
                    "--dev-requirements",
                    str(dev_requirements),
                    "--pyproject",
                    str(pyproject),
                    "--readme",
                    str(readme),
                    "--dev",
                ]),
                patch.object(install_env, "current_python_is_supported", return_value=True),
                patch.object(install_env, "sync_pyproject"),
                patch.object(install_env, "sync_readme"),
                patch.object(install_env.venv.EnvBuilder, "create"),
                patch.object(install_env, "venv_python_path", return_value=venv_python),
                patch.object(install_env, "run_command", side_effect=lambda command, dry_run: commands.append(command)),
            ):
                exit_code = install_env.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(commands[0], [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        self.assertEqual(commands[1], [str(venv_python), "-m", "pip", "install", "-r", str(runtime_requirements)])
        self.assertEqual(commands[2], [str(venv_python), "-m", "pip", "install", "-r", str(dev_requirements)])
        self.assertEqual(commands[3], [str(venv_python), "-m", "pip", "install", "-e", str(install_env.REPO_ROOT), "--no-deps"])