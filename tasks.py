from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path

from invoke import task


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _shell_command(*parts: str | Path) -> str:
    return subprocess.list2cmdline([str(part) for part in parts])


def _run_script(context, script_name: str, *args: str) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    context.run(_shell_command(PYTHON, script_path, *args), pty=False)


@task
def install(context, dry_run: bool = False, dev: bool = False) -> None:
    """Create the supported virtual environment and install the package."""
    args: list[str] = []
    if dry_run:
        args.append("--dry-run")
    if dev:
        args.append("--dev")
    _run_script(context, "install_env.py", *args)


@task(name="sync-deps")
def sync_deps(context, check: bool = False) -> None:
    """Sync dependency metadata from requirements/runtime.txt."""
    args = ["--check"] if check else []
    _run_script(context, "sync_dependencies.py", *args)


@task(name="backfill-dist")
def backfill_dist(context, dry_run: bool = False, fetch_tags: bool = False) -> None:
    """Build missing distribution artifacts for tagged releases."""
    args: list[str] = []
    if dry_run:
        args.append("--dry-run")
    if fetch_tags:
        args.append("--fetch-tags")
    _run_script(context, "backfill_dist.py", *args)


@task
def docs(context, clean: bool = True, strict: bool = True) -> None:
    """Build the Sphinx documentation into docs/build/html."""
    docs_source = REPO_ROOT / "docs" / "source"
    docs_build = REPO_ROOT / "docs" / "build"
    html_output = docs_build / "html"

    if clean and docs_build.exists():
        shutil.rmtree(docs_build)

    command_parts: list[str | Path] = [PYTHON, "-m", "sphinx"]
    if strict:
        command_parts.append("-W")
    command_parts.extend(["-b", "html", docs_source, html_output])
    context.run(_shell_command(*command_parts), pty=False)