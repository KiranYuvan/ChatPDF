from app.chunker import chunk_pages
from app.pdf_loader import PageText

LONG_TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))


def test_chunks_respect_max_size() -> None:
    chunks = chunk_pages([PageText(1, LONG_TEXT)], "doc.pdf", chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_pages([PageText(1, LONG_TEXT)], "doc.pdf", chunk_size=300, chunk_overlap=80)
    first, second = chunks[0].text, chunks[1].text
    # The beginning of chunk 2 repeats text from the end of chunk 1.
    assert second[:20] in first


def test_metadata_and_deterministic_ids() -> None:
    pages = [PageText(1, "Short page one."), PageText(3, "Short page three.")]
    chunks = chunk_pages(pages, "doc.pdf", chunk_size=300, chunk_overlap=50)
    assert [(c.page, c.source) for c in chunks] == [(1, "doc.pdf"), (3, "doc.pdf")]
    assert [c.id for c in chunks] == ["doc.pdf::p1::c0", "doc.pdf::p3::c0"]
    # Same input -> same ids, so re-indexing overwrites rather than duplicates.
    assert chunks == chunk_pages(pages, "doc.pdf", chunk_size=300, chunk_overlap=50)


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_pages([PageText(1, "Tiny.")], "doc.pdf", chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 1 and chunks[0].text == "Tiny."
