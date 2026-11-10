"""PDF -> text, one record per page (page numbers are kept so answers can cite them)."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    """Text of one PDF page. `page_number` is 1-based, as a human would cite it."""

    page_number: int
    text: str


def load_pdf(source: str | Path | bytes) -> list[PageText]:
    """Extract text from every non-empty page of a PDF given as a path or raw bytes (an upload).

    Raises:
        ValueError: if no page has extractable text (typically a scanned, image-only PDF).
    """
    reader = PdfReader(BytesIO(source) if isinstance(source, bytes) else source)
    pages = [
        PageText(page_number=i, text=text)
        for i, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]
    if not pages:
        raise ValueError("No extractable text found. The PDF may be scanned (image-only) and needs OCR.")
    return pages
