"""Split page text into overlapping chunks, keeping source + page metadata for citations."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.pdf_loader import PageText


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage. `id` is deterministic so re-indexing overwrites instead of duplicating."""

    id: str
    text: str
    source: str
    page: int


def chunk_pages(pages: list[PageText], source: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split each page into chunks of at most `chunk_size` characters with `chunk_overlap` shared.

    "Recursive" splitting tries paragraph breaks first, then lines, sentences, words, so chunks
    end at natural boundaries. Chunks never span pages, so every chunk has exactly one page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        Chunk(id=f"{source}::p{page.page_number}::c{i}", text=piece, source=source, page=page.page_number)
        for page in pages
        for i, piece in enumerate(splitter.split_text(page.text))
    ]
