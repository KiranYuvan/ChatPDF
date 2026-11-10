import pytest

from app.pdf_loader import load_pdf
from tests.conftest import make_pdf


def test_extracts_text_per_page_with_1_based_numbers() -> None:
    pages = load_pdf(make_pdf(["Alpha page text", "Beta page text"]))
    assert [p.page_number for p in pages] == [1, 2]
    assert "Alpha" in pages[0].text
    assert "Beta" in pages[1].text


def test_blank_pages_are_skipped_but_numbering_is_kept() -> None:
    pages = load_pdf(make_pdf(["First", "", "Third"]))
    assert [p.page_number for p in pages] == [1, 3]


def test_non_pdf_bytes_raise_friendly_error() -> None:
    with pytest.raises(ValueError, match="Could not read this file as a PDF"):
        load_pdf(b"this is not a pdf")


def test_pdf_without_text_raises() -> None:
    with pytest.raises(ValueError, match="scanned"):
        load_pdf(make_pdf(["", ""]))
