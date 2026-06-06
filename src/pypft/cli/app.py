from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pypft.core.exceptions import PyPFTError
from pypft.workflows import (
    TransformWorkflowRequest,
    compare_field_files,
    inspect_trace_source,
    render_field_file,
    render_trace_source,
    run_transform_workflow,
    validate_roundtrip,
)


class ComplexViewOption(str, Enum):
    magnitude = "magnitude"
    phase = "phase"
    angular = "angular"
    both = "both"


class FieldKindOption(str, Enum):
    field = "field"
    spatial_samples = "spatial_samples"
    angular_spectrum = "angular_spectrum"
    radial_spectrum = "radial_spectrum"
    frequency_samples = "frequency_samples"


app = typer.Typer(no_args_is_help=True)
transform_app = typer.Typer(no_args_is_help=True)
visualize_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
validate_app = typer.Typer(no_args_is_help=True)
app.add_typer(transform_app, name="transform")
app.add_typer(visualize_app, name="visualize")
app.add_typer(inspect_app, name="inspect")
app.add_typer(validate_app, name="validate")

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
    complex_view: ComplexViewOption = typer.Option(
        ComplexViewOption.both,
        "--complex-view",
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
    complex_view: ComplexViewOption = typer.Option(
        ComplexViewOption.both,
        "--complex-view",
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
    complex_view: ComplexViewOption,
    save_all_views: bool,
    save_stage_arrays: bool,
    backend: str | None,
    dht_implementation: str | None,
) -> None:
    try:
        request = TransformWorkflowRequest(
            direction=direction,
            input_path=input_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            gamma=gamma,
            complex_view=complex_view.value,
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
        summary.add_row(
            "Saved stage arrays",
            str(len(result.stage_array_paths)),
        )
    if result.figure_paths:
        summary.add_row("Saved figures", str(len(result.figure_paths)))
    _console.print(summary)


@visualize_app.command("field")
def visualize_field(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        resolve_path=True,
    ),
    field_kind: FieldKindOption = typer.Option(
        FieldKindOption.field,
        "--field-kind",
    ),
    gamma: float = typer.Option(1.0, "--gamma"),
    complex_view: ComplexViewOption = typer.Option(
        ComplexViewOption.both,
        "--complex-view",
    ),
) -> None:
    try:
        figure_paths = render_field_file(
            input_path,
            output_dir,
            field_kind=field_kind.value,
            gamma=gamma,
            complex_view=complex_view.value,
        )
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    summary = Table(show_header=False, box=None)
    summary.add_row("Source field", str(input_path))
    summary.add_row("Output directory", str(output_dir))
    summary.add_row("Saved figures", str(len(figure_paths)))
    _console.print(summary)


@visualize_app.command("trace")
def visualize_trace(
    source_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        resolve_path=True,
    ),
    gamma: float | None = typer.Option(None, "--gamma"),
    complex_view: ComplexViewOption | None = typer.Option(
        None,
        "--complex-view",
    ),
) -> None:
    try:
        figure_paths = render_trace_source(
            source_path,
            output_dir,
            gamma=gamma,
            complex_view=(
                complex_view.value if complex_view is not None else None
            ),
        )
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    summary = Table(show_header=False, box=None)
    summary.add_row("Trace source", str(source_path))
    summary.add_row("Output directory", str(output_dir))
    summary.add_row("Rendered stages", str(len(figure_paths)))
    summary.add_row(
        "Saved figures",
        str(sum(len(paths) for paths in figure_paths.values())),
    )
    _console.print(summary)


@inspect_app.command("trace")
def inspect_trace(
    source_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    try:
        inspection = inspect_trace_source(source_path)
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    summary = Table(show_header=False, box=None)
    summary.add_row("Source kind", inspection.source_kind)
    summary.add_row("Direction", inspection.direction)
    summary.add_row("Trace path", str(inspection.trace_path))
    if inspection.input_path is not None:
        summary.add_row("Input path", str(inspection.input_path))
    if inspection.metadata_path is not None:
        summary.add_row("Metadata path", str(inspection.metadata_path))
    if inspection.backend is not None:
        summary.add_row("Backend", inspection.backend)
    if inspection.dht_implementation is not None:
        summary.add_row(
            "DHT implementation",
            inspection.dht_implementation,
        )
    if inspection.gamma is not None:
        summary.add_row("Gamma", f"{inspection.gamma:g}")
    if inspection.complex_view is not None:
        summary.add_row("Complex view", inspection.complex_view)
    if inspection.package_version is not None:
        summary.add_row("Package version", inspection.package_version)
    summary.add_row(
        "Stage names",
        ", ".join(frame.stage for frame in inspection.frames),
    )
    _console.print(summary)

    frames = Table(
        "#",
        "Stage",
        "Field Kind",
        "Shape",
        "Dtype",
        "Array",
        "Figures",
    )
    for frame in inspection.frames:
        frames.add_row(
            str(frame.index),
            frame.stage,
            frame.field_kind,
            str(frame.shape),
            frame.dtype,
            "yes" if frame.has_array else "no",
            str(frame.figure_count),
        )
    _console.print(frames)


@validate_app.command("compare")
def validate_compare(
    reference_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    candidate_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        resolve_path=True,
    ),
    field_kind: FieldKindOption = typer.Option(
        FieldKindOption.field,
        "--field-kind",
    ),
    gamma: float = typer.Option(1.0, "--gamma"),
    complex_view: ComplexViewOption = typer.Option(
        ComplexViewOption.both,
        "--complex-view",
    ),
    atol: float = typer.Option(1e-10, "--atol"),
    rtol: float = typer.Option(1e-10, "--rtol"),
) -> None:
    try:
        result = compare_field_files(
            reference_path,
            candidate_path,
            output_dir,
            field_kind=field_kind.value,
            gamma=gamma,
            complex_view=complex_view.value,
            atol=atol,
            rtol=rtol,
        )
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _print_validation_summary(
        label="Comparison report",
        report_path=result.report_path,
        figure_paths=result.figure_paths,
        metrics=result.metrics,
        extra_rows=(
            ("Reference", str(reference_path)),
            ("Candidate", str(candidate_path)),
            ("Output directory", str(output_dir)),
        ),
    )


@validate_app.command("roundtrip")
def validate_roundtrip_command(
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
    complex_view: ComplexViewOption = typer.Option(
        ComplexViewOption.both,
        "--complex-view",
    ),
    atol: float = typer.Option(1e-10, "--atol"),
    rtol: float = typer.Option(1e-10, "--rtol"),
    backend: str | None = typer.Option(None, "--backend"),
    dht_implementation: str | None = typer.Option(
        None,
        "--dht-implementation",
    ),
) -> None:
    try:
        result = validate_roundtrip(
            input_path,
            output_dir,
            metadata_path=metadata_path,
            gamma=gamma,
            complex_view=complex_view.value,
            atol=atol,
            rtol=rtol,
            backend=backend,
            dht_implementation=dht_implementation,
        )
    except (PyPFTError, ValueError) as error:
        _console.print(f"[bold red]Error:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    _print_validation_summary(
        label="Roundtrip report",
        report_path=result.report_path,
        figure_paths=result.figure_paths,
        metrics=result.metrics,
        extra_rows=(
            ("Input", str(input_path)),
            ("Forward array", str(result.forward_path)),
            ("Reconstruction", str(result.reconstruction_path)),
            ("Output directory", str(output_dir)),
        ),
    )


def _print_validation_summary(
    *,
    label: str,
    report_path: Path,
    figure_paths: tuple[Path, ...],
    metrics,
    extra_rows: tuple[tuple[str, str], ...],
) -> None:
    summary = Table(show_header=False, box=None)
    for key, value in extra_rows:
        summary.add_row(key, value)
    summary.add_row(label, str(report_path))
    summary.add_row("Saved figures", str(len(figure_paths)))
    summary.add_row(
        "Status",
        "pass" if metrics.passes_tolerance else "fail",
    )
    summary.add_row("Max abs error", f"{metrics.max_abs_error:.6g}")
    summary.add_row("RMSE", f"{metrics.rmse:.6g}")
    summary.add_row(
        "Relative L2 error",
        f"{metrics.relative_l2_error:.6g}",
    )
    summary.add_row(
        "Tolerance",
        f"atol={metrics.atol:.3g}, rtol={metrics.rtol:.3g}",
    )
    _console.print(summary)


__all__ = ["app"]
