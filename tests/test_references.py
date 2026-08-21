"""Tests for the citation machinery (``pypft.references``)."""

import pytest

from pypft.references import Reference, bibliography, cite


def test_cite_joins_inline_labels_in_order():
    """``cite`` joins the given references' inline labels, order preserved."""
    result = cite(Reference.YAO_BADDOUR_2020_PFT_PART2, Reference.BADDOUR_2019_DHT)
    assert result == "[YaoBaddour2020], [Baddour2019a]"


def test_bibliography_sorts_by_key_and_includes_the_doi():
    """``bibliography`` sorts its entries by key and renders each one's DOI."""
    result = bibliography(
        Reference.YAO_BADDOUR_2020_PFT_PART2, Reference.BADDOUR_2019_DHT
    )
    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("- [Baddour2019a]")
    assert lines[1].startswith("- [YaoBaddour2020]")
    assert "doi:10.5772/intechopen.84399" in lines[0]


def test_bibliography_of_every_reference_has_no_duplicate_keys():
    """Every ``Reference`` member has a distinct citation key."""
    keys = [member.value.key for member in Reference]
    assert len(keys) == len(set(keys))


def test_cite_rejects_a_non_reference_argument():
    """``cite`` raises rather than silently accepting a non-``Reference``."""
    with pytest.raises(TypeError):
        cite("Baddour2019a")  # type: ignore[arg-type]
