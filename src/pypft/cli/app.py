from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pypft.cli.commands import build_transform_request, validate_complex_view
from pypft.core.exceptions import PyPFTError
from pypft.workflows import run_transform_workflow

app = typer.Typer(no_args_is_help=True)
transform_app = typer.Typer(no_args_is_help=True)
app.add_typer(transform_app, name="transform")

_console = Console(stderr=True)


@transform_app.command("forward")
def transform_forward(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    metadata_path: Path | None = typer.Option(
        None,
        "--metadata",
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        resolve_path=True,
    ),
    gamma: float = typer.Option(1.0, "--gamma"),
    complex_view: str = typer.Option(
        "both",
        "--complex-view",
        callback=lambda value: validate_complex_view(value),
    ),
    save_all_views: bool = typer.Option(False, "--save-all-views"),
    save_stage_arrays: bool = typer.Option(False, "--save-stage-arrays"),
    backend: str | None = typer.Option(None, "--backend"),
    dht_implementation: str | None = typer.Option(
        None,
        "--dht-implementation",
    ),
) -> None:
    _run_transform_command(
        direction="forward",
        input_path=input_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        gamma=gamma,
        complex_view=complex_view,
        save_all_views=save_all_views,
        save_stage_arrays=save_stage_arrays,
        backend=backend,
        dht_implementation=dht_implementation,
    )


@transform_app.command("backward")
def transform_backward(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    metadata_path: Path | None = typer.Option(
        None,
        "--metadata",
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        resolve_path=True,
    ),
    gamma: float = typer.Option(1.0, "--gamma"),
    complex_view: str = typer.Option(
        "both",
        "--complex-view",
        callback=lambda value: validate_complex_view(value),
    ),
    save_all_views: bool = typer.Option(False, "--save-all-views"),
    save_stage_arrays: bool = typer.Option(False, "--save-stage-arrays"),
    backend: str | None = typer.Option(None, "--backend"),
    dht_implementation: str | None = typer.Option(
        None,
        "--dht-implementation",
    ),
) -> None:
    _run_transform_command(
        direction="backward",
        input_path=input_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        gamma=gamma,
        complex_view=complex_view,
        save_all_views=save_all_views,
        save_stage_arrays=save_stage_arrays,
        backend=backend,
        dht_implementation=dht_implementation,
    )


def _run_transform_command(
    *,
    direction: str,
    input_path: Path,
    metadata_path: Path | None,
    output_dir: Path,
    gamma: float,
    complex_view: str,
    save_all_views: bool,
    save_stage_arrays: bool,
    backend: str | None,
    dht_implementation: str | None,
) -> None:
    try:
        request = build_transform_request(
            direction=direction,
            input_path=input_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            gamma=gamma,
            complex_view=complex_view,
            save_all_views=save_all_views,
            save_stage_arrays=save_stage_arrays,
            backend=backend,
            dht_implementation=dht_implementation,
        )
        result = run_transform_workflow(request)
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    summary = Table(show_header=False, box=None)
    summary.add_row("Output array", str(result.output_path))
    summary.add_row("Trace", str(result.trace_path))
    summary.add_row("Manifest", str(result.manifest_path))
    summary.add_row("Trace stages", str(len(result.trace.frames)))
    if result.stage_array_paths:
        summary.add_row("Saved stage arrays", str(len(result.stage_array_paths)))
    if result.figure_paths:
        summary.add_row("Saved figures", str(len(result.figure_paths)))
    _console.print(summary)


__all__ = ["app"]
