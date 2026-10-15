"""PyPFT's terminal interface: ``python -m pypft``, or the ``pypft`` command.

The app loads a polar array from a file, runs the PFT or the IPFT on it, and views or
saves the result's figures, one screen per step (see ``pypft.cli._app``). It is a thin
front end over the public API and adds no numerics: everything it does is a few calls
of ``pypft.grid``, ``pypft.transform``, ``pypft.ring`` and ``pypft.viz``, so a script
does the same without it. It runs in any terminal, with no graphical toolkit; a figure
opens in a matplotlib window when matplotlib has a backend that opens one, and is
written to a temporary file otherwise.

The file handling is public too, for scripts: ``load_array`` reads the polar array
files (``POLAR_FILE_SUFFIXES``), ``acquisition_input`` converts a selection of an
array of radial lines (from any of ``LINE_FILE_SUFFIXES``, read with
``pypft.acquisition.load_stored_array``) into uniform polar k-space,
``output_filename`` names a saved figure, and
``save_signal_figures`` saves a signal's figures under those names, warning with
``CLIPerformanceWarning`` above ``SLOW_BATCH_SIZE`` batch elements.

This package is not imported by ``pypft`` itself, since it needs ``textual``.
"""

import argparse

from ._app import TUTORIALS_URL, PyPFTApp
from ._io import (
    LINE_FILE_SUFFIXES,
    POLAR_FILE_SUFFIXES,
    SLOW_BATCH_SIZE,
    SUPPORTED_SUFFIXES,
    CLIPerformanceWarning,
    FigurePart,
    LoadedInput,
    acquisition_input,
    load_array,
    output_filename,
    save_signal_figures,
)

__all__ = [
    "CLIPerformanceWarning",
    "FigurePart",
    "LINE_FILE_SUFFIXES",
    "LoadedInput",
    "POLAR_FILE_SUFFIXES",
    "PyPFTApp",
    "SLOW_BATCH_SIZE",
    "SUPPORTED_SUFFIXES",
    "TUTORIALS_URL",
    "acquisition_input",
    "load_array",
    "main",
    "output_filename",
    "save_signal_figures",
]

#: The name the app is started by, shown in ``--help``.
_PROGRAM = "pypft"


def main(argv: list[str] | None = None) -> int:
    """Run the terminal interface from the command line.

    :param argv: The command-line arguments; ``sys.argv[1:]`` when omitted. Only
        ``--help`` is accepted.
    :type argv: list[str] | None
    :returns: The exit code, ``0`` when the user quits.
    :rtype: int

    """
    parser = argparse.ArgumentParser(
        prog=_PROGRAM,
        description=(
            "Load a polar array, transform it with the polar Fourier transform, and "
            "view or save its figures, in a terminal interface."
        ),
    )
    parser.parse_args(args=argv)
    code = PyPFTApp().run()
    return 0 if code is None else code
