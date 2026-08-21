"""Run bench_dft.py under pytest-benchmark and export results sorted by speed.

Usage: ``uv run python benchmarks/run_dft_benchmarks.py``

Writes a Markdown report to ``.local_files/benchmarks/results/`` (gitignored,
since reports are generated local artifacts, not source) and prints the
fastest ``DFTImplementation`` for the "forward, repeated calls" scenario --
the one that should be hardcoded as ``pypft.dft.DEFAULT_IMPLEMENTATION``.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = REPO_ROOT / ".local_files" / "benchmarks" / "results"


def _run_benchmarks(json_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(BENCH_DIR / "bench_dft.py"),
            "--benchmark-only",
            f"--benchmark-json={json_path}",
            "-q",
        ],
        check=True,
    )


def _test_group(name: str) -> str:
    return name.split("[", 1)[0]


def _format_table(benchmarks: list[dict]) -> str:
    lines = ["| Test | Mean (s) | Stddev (s) |", "| --- | --- | --- |"]
    for bench in sorted(benchmarks, key=lambda b: b["stats"]["mean"]):
        stats = bench["stats"]
        lines.append(
            f"| `{bench['name']}` | {stats['mean']:.3e} | {stats['stddev']:.3e} |"
        )
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "_raw_dft_benchmark.json"
    _run_benchmarks(json_path)

    data = json.loads(json_path.read_text())
    benchmarks = data["benchmarks"]

    groups: dict[str, list[dict]] = {}
    for bench in benchmarks:
        groups.setdefault(_test_group(bench["name"]), []).append(bench)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    report_lines = [f"# DFT benchmark results ({timestamp})", ""]
    for group_name, group_benchmarks in groups.items():
        report_lines.append(f"## {group_name}")
        report_lines.append("")
        report_lines.append(_format_table(group_benchmarks))
        report_lines.append("")

    forward_group = groups.get("test_bench_forward_repeated", [])
    if forward_group:
        winner = min(forward_group, key=lambda b: b["stats"]["mean"])
        report_lines.append(
            f"Fastest for repeated forward calls: `{winner['name']}` "
            f"(mean {winner['stats']['mean']:.3e}s) -- this is the recommended "
            "DEFAULT_IMPLEMENTATION."
        )
        print(f"Winner: {winner['name']} (mean {winner['stats']['mean']:.3e}s)")

    report_path = RESULTS_DIR / f"report_{timestamp}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
