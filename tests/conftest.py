"""Shared pytest fixtures. pytest discovers this file automatically (like xUnit fixtures)."""

from pathlib import Path

import pytest
from fpdf import FPDF

from app.config import Settings


def make_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF with one page per string (empty string = blank page)."""
    pdf = FPDF()
    for text in pages:
        pdf.add_page()
        if text:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings: ignores the developer's .env and uses a throwaway Chroma folder."""
    return Settings(_env_file=None, chroma_dir=tmp_path / "chroma", pdf_dir=tmp_path / "pdfs")


class FakeLLM:
    """Test double for LLMClient: records the prompt and returns a canned answer (like a Moq mock)."""

    name = "fake"

    def __init__(self, reply: str = "fake answer") -> None:
        self.reply = reply
        self.last_user_prompt: str | None = None

    def chat(self, system: str, user: str) -> str:
        self.last_user_prompt = user
        return self.reply


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
