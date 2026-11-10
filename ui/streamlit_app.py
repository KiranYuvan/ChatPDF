"""Streamlit UI: upload a PDF, index it, and chat with it.

Run from the project root:
    streamlit run streamlit_app.py        # root entry point (also used by Streamlit Cloud)
    streamlit run ui/streamlit_app.py     # works too, after `pip install -e .`

Each browser session gets its own namespace, so on a shared deployment visitors only ever
see and query their own uploads.
"""

import uuid

import streamlit as st

from app.config import get_settings
from app.llm import LLMError
from app.logging_config import configure_logging
from app.rag import Answer, RAGService

st.set_page_config(page_title="ChatPDF", page_icon="📄")


@st.cache_resource(show_spinner="Loading models…")
def get_service() -> RAGService:
    """One RAGService per server process (singleton lifetime), not rebuilt on every rerun."""
    configure_logging(get_settings().log_level)
    return RAGService()


def session_namespace() -> str:
    """A random id per browser session (like a per-user tenant id), created on first visit."""
    return st.session_state.setdefault("namespace", uuid.uuid4().hex)


def render_sidebar(service: RAGService, namespace: str) -> str | None:
    """Upload + index controls and the document picker. Returns the active document name."""
    with st.sidebar:
        st.header("1. Upload a PDF")
        uploaded = st.file_uploader(f"PDF file (max {service.settings.max_upload_mb} MB)", type="pdf")
        if uploaded and st.button("Index PDF", type="primary"):
            with st.spinner("Extracting, chunking and embedding…"):
                try:
                    result = service.index_pdf(uploaded.getvalue(), uploaded.name, namespace)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.active_doc = result.source
                    st.session_state.messages = []
                    st.success(f"Indexed **{result.source}**: {result.pages} pages → {result.chunks} chunks")

        docs = service.documents(namespace)
        if not docs:
            return None
        st.header("2. Pick a document")
        current = st.session_state.get("active_doc")
        st.session_state.active_doc = st.selectbox(
            "Your documents", docs, index=docs.index(current) if current in docs else 0
        )
        if st.button("Clear my documents"):
            service.clear(namespace)
            st.session_state.pop("active_doc", None)
            st.session_state.messages = []
            st.rerun()
        st.caption("Documents are private to this browser session and are removed when the app restarts.")
        return st.session_state.active_doc


def render_sources(answer: Answer) -> None:
    """Show which chunks the answer was grounded on, with page and similarity score."""
    with st.expander(f"Sources ({len(answer.sources)} chunks) · model: {answer.model}"):
        for hit in answer.sources:
            st.markdown(f"**Page {hit.page}** · similarity {hit.score:.2f}")
            st.text(hit.text)


def main() -> None:
    st.title("📄 Chat with your PDF")
    st.caption("Free & local RAG: pypdf → MiniLM embeddings → Chroma → local / free-tier LLM")

    service = get_service()
    namespace = session_namespace()
    active = render_sidebar(service, namespace)
    if not active:
        st.info("Upload a PDF in the sidebar and click **Index PDF** to start.")
        return

    st.write(f"Chatting with **{active}**")
    messages: list[dict[str, str]] = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the PDF"):
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating… (CPU models can take 10-60 s)"):
                try:
                    answer = service.ask(question, source=active, namespace=namespace)
                except LLMError as exc:
                    st.error(str(exc))
                    return
            st.markdown(answer.text)
            render_sources(answer)
        messages.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":  # `streamlit run` executes the script as __main__
    main()
