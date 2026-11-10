"""Streamlit UI: upload a PDF, index it, and chat with it.

Run from the project root (after `pip install -e .`):
    streamlit run ui/streamlit_app.py
"""

import streamlit as st

from app.rag import Answer, RAGService

st.set_page_config(page_title="ChatPDF", page_icon="📄")


@st.cache_resource(show_spinner="Loading models…")
def get_service() -> RAGService:
    """One RAGService per server process (singleton lifetime), not rebuilt on every rerun."""
    return RAGService()


def render_sidebar(service: RAGService) -> str | None:
    """Upload + index controls and the document picker. Returns the active document name."""
    with st.sidebar:
        st.header("1. Upload a PDF")
        uploaded = st.file_uploader("PDF file", type="pdf")
        if uploaded and st.button("Index PDF", type="primary"):
            with st.spinner("Extracting, chunking and embedding…"):
                try:
                    result = service.index_pdf(uploaded.getvalue(), uploaded.name)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.active_doc = result.source
                    st.session_state.messages = []
                    st.success(f"Indexed **{result.source}**: {result.pages} pages → {result.chunks} chunks")

        docs = service.store.sources()
        if not docs:
            return None
        st.header("2. Pick a document")
        current = st.session_state.get("active_doc")
        st.session_state.active_doc = st.selectbox(
            "Indexed documents", docs, index=docs.index(current) if current in docs else 0
        )
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
    active = render_sidebar(service)
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
                answer = service.ask(question, source=active)
            st.markdown(answer.text)
            render_sources(answer)
        messages.append({"role": "assistant", "content": answer.text})


main()
