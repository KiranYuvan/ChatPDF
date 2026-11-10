"""End-to-end demo without the UI: index a PDF, then ask a few questions.

python -m scripts.demo [path/to/file.pdf]     # default: data/pdfs/sample_handbook.pdf
"""

import sys
import time
from pathlib import Path

from app.config import get_settings
from app.logging_config import configure_logging
from app.rag import RAGService

QUESTIONS = [
    "How many days of paid annual leave do employees get, and how many can be carried over?",
    "What is the hotel cap per night when travelling?",
    "How quickly must a lost laptop be reported?",
    "What is the company's stock price?",  # not in the PDF -> should say so instead of guessing
]


def main() -> int:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else get_settings().pdf_dir / "sample_handbook.pdf"
    if not pdf.exists():
        print(f"{pdf} not found. Create the sample with:  python -m scripts.make_sample_pdf")
        return 1

    configure_logging("WARNING")  # keep the demo output clean
    service = RAGService()
    start = time.perf_counter()
    result = service.index_pdf(pdf.read_bytes(), pdf.name)
    print(
        f"Indexed {result.source}: {result.pages} pages -> {result.chunks} chunks "
        f"({time.perf_counter() - start:.1f}s)\n"
    )

    for question in QUESTIONS:
        start = time.perf_counter()
        answer = service.ask(question, source=result.source)
        hits = ", ".join(f"p{h.page}({h.score:.2f})" for h in answer.sources)
        print(f"Q: {question}\nA: {answer.text}")
        print(f"   [{answer.model}, {time.perf_counter() - start:.1f}s] retrieved: {hits}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
