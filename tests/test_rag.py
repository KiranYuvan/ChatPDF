"""RAGService end to end with a fake LLM: real retrieval, no slow model download for generation."""

import pytest

from app.config import Settings
from app.llm import LLMError
from app.rag import EMPTY_QUESTION, NO_DOCUMENTS, RAGService
from tests.conftest import FakeLLM, make_pdf

pytestmark = pytest.mark.integration

HANDBOOK = make_pdf(["Paid annual leave is 24 days per year.", "Hotels are capped at 220 dollars per night."])


class FailingLLM:
    name = "failing"

    def chat(self, system: str, user: str) -> str:
        raise TimeoutError("upstream timed out")


def test_ask_before_indexing_returns_guidance(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)
    assert service.ask("anything").text == NO_DOCUMENTS
    assert service.ask("   ").text == EMPTY_QUESTION
    assert fake_llm.last_user_prompt is None  # LLM never called


def test_index_then_ask_grounds_prompt_in_retrieved_pages(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)

    result = service.index_pdf(HANDBOOK, "handbook.pdf")
    answer = service.ask("What is the hotel limit?", source="handbook.pdf")

    assert (result.pages, result.chunks) == (2, 2)
    assert answer.text == "fake answer" and answer.model == "fake"
    assert answer.sources[0].page == 2
    assert fake_llm.last_user_prompt is not None
    assert "[page 2]" in fake_llm.last_user_prompt and "What is the hotel limit?" in fake_llm.last_user_prompt


def test_filename_paths_are_stripped(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)
    assert service.index_pdf(HANDBOOK, "../../etc/handbook.pdf").source == "handbook.pdf"


def test_upload_size_limit(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings.model_copy(update={"max_upload_mb": 1}), llm=fake_llm)
    with pytest.raises(ValueError, match="limit is 1 MB"):
        service.index_pdf(b"0" * (2 * 1024 * 1024), "big.pdf")


def test_llm_failure_becomes_friendly_error(settings: Settings) -> None:
    service = RAGService(settings, llm=FailingLLM())
    service.index_pdf(HANDBOOK, "handbook.pdf")
    with pytest.raises(LLMError, match="try again"):
        service.ask("What is the hotel limit?")


def test_users_only_see_their_own_documents(settings: Settings, fake_llm: FakeLLM) -> None:
    service = RAGService(settings, llm=fake_llm)
    service.index_pdf(HANDBOOK, "handbook.pdf", namespace="alice")

    assert service.documents("alice") == ["handbook.pdf"]
    assert service.documents("bob") == []
    assert service.ask("hotel?", namespace="bob").text == NO_DOCUMENTS
