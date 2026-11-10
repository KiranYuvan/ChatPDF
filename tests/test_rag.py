"""RAGService end to end with a fake LLM: real retrieval, no slow model download for generation."""

import pytest

from app.config import Settings
from app.rag import NO_DOCUMENTS, RAGService
from tests.conftest import FakeLLM, make_pdf

pytestmark = pytest.mark.integration


def test_ask_before_indexing_returns_guidance(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)
    assert service.ask("anything").text == NO_DOCUMENTS
    assert fake_llm.last_user_prompt is None  # LLM never called


def test_index_then_ask_grounds_prompt_in_retrieved_pages(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)
    pdf = make_pdf(["Paid annual leave is 24 days per year.", "Hotels are capped at 220 dollars per night."])

    result = service.index_pdf(pdf, "handbook.pdf")
    answer = service.ask("What is the hotel limit?", source="handbook.pdf")

    assert (result.pages, result.chunks) == (2, 2)
    assert answer.text == "fake answer" and answer.model == "fake"
    assert answer.sources[0].page == 2
    assert fake_llm.last_user_prompt is not None
    assert "[page 2]" in fake_llm.last_user_prompt and "What is the hotel limit?" in fake_llm.last_user_prompt
