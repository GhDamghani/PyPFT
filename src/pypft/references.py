"""Citation machinery for PyPFT's tutorials and API documentation.

Notebook Markdown cells cannot execute code, so a citing sentence writes a
``Reference`` member's ``inline`` label out literally (e.g. "...is
self-inverse [Baddour2019a, Eq. 41]."); only the notebook's final cell -- a
code cell rendering ``bibliography(...)`` -- is generated from this module.
This keeps every citing sentence readable on GitHub, in nbviewer, and in any
unexecuted checkout, while still machine-checking that every label resolves
to a real reference and that every citing notebook's bibliography cell lists
exactly what it cited (``tests/test_notebook_citations.py``).
"""

from dataclasses import dataclass
from enum import Enum

from pypft.utils.validators import EnumValidator


@dataclass(frozen=True)
class _Entry:
    """One citable work.

    :param key: The citation label used in prose, without the brackets
        (e.g. ``"Baddour2019a"``).
    :type key: str
    :param inline: The literal label a notebook's Markdown cell writes out
        (e.g. ``"[Baddour2019a]"``).
    :type inline: str
    :param full: The complete, human-readable reference string.
    :type full: str
    :param doi: The work's DOI, without a resolver prefix, if known.
    :type doi: str

    """

    key: str
    inline: str
    full: str
    doi: str = ""


class Reference(Enum):
    """The scientific sources PyPFT's math and API docs cite."""

    BADDOUR_2019_DHT = _Entry(
        key="Baddour2019a",
        inline="[Baddour2019a]",
        full=(
            'N. Baddour, "The Discrete Hankel Transform," in Fourier '
            "Transforms - Century of Digitalization and Increasing "
            "Expectations, IntechOpen, 2019."
        ),
        doi="10.5772/intechopen.84399",
    )
    BADDOUR_2019_PFT_PART1 = _Entry(
        key="Baddour2019b",
        inline="[Baddour2019b]",
        full=(
            'N. Baddour, "Discrete Two-Dimensional Fourier Transform in '
            'Polar Coordinates Part I: Theory and Operational Rules," '
            "Mathematics, 7(8):698, 2019."
        ),
        doi="10.3390/math7080698",
    )
    YAO_BADDOUR_2020_PFT_PART2 = _Entry(
        key="YaoBaddour2020",
        inline="[YaoBaddour2020]",
        full=(
            'X. Yao and N. Baddour, "Discrete Two-Dimensional Fourier '
            "Transform in Polar Coordinates Part II: Numerical Computation "
            'and Approximation of the Continuous Transform," PeerJ '
            "Computer Science, 6:e257, 2020."
        ),
        doi="10.7717/peerj-cs.257",
    )
    GOLSHANI_2017_MRM = _Entry(
        key="Golshani2017",
        inline="[Golshani2017]",
        full=(
            'S. Golshani and A. Nasiraei-Moghaddam, "Efficient Radial '
            "Tagging CMR Exam: A Coherent k-Space Reading and Image "
            'Reconstruction Approach," Magnetic Resonance in Medicine, '
            "77(4):1459-1472, 2017."
        ),
        doi="10.1002/mrm.26219",
    )


def _validate_references(refs: tuple[Reference, ...]) -> None:
    """Validate a tuple of ``Reference`` arguments.

    :param refs: The references to validate.
    :type refs: tuple[Reference, ...]
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument is not a ``Reference`` member.

    """
    for ref in refs:
        EnumValidator.type_is_enum(value=ref)
        EnumValidator.value_is_enum_member(value=ref, enum_class=Reference)


def cite(*refs: Reference) -> str:
    """Join one or more references' inline labels, in the order given.

    :param refs: The references to cite.
    :type refs: Reference
    :returns: The references' inline labels, joined by ``", "``.
    :rtype: str
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument is not a ``Reference`` member.

    """
    _validate_references(refs)
    return ", ".join(ref.value.inline for ref in refs)


def bibliography(*refs: Reference) -> str:
    """Render a sorted Markdown reference list from the given references.

    :param refs: The references to list.
    :type refs: Reference
    :returns: A Markdown bullet list, one entry per reference, sorted by key.
    :rtype: str
    :raises TypeError: If any argument has the wrong type.
    :raises ValueError: If any argument is not a ``Reference`` member.

    """
    _validate_references(refs)
    lines = []
    for ref in sorted(refs, key=lambda ref: ref.value.key):
        entry = ref.value
        line = f"- {entry.inline} {entry.full}"
        if entry.doi:
            line += f" doi:{entry.doi}"
        lines.append(line)
    return "\n".join(lines)
