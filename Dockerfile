# ChatPDF API image (FastAPI + uvicorn). CPU-only torch; ~3.4 GB uncompressed (torch ~0.8 GB of it).
#   docker build -t chatpdf-api .
#   docker run -p 8000:8000 -e GROQ_API_KEY=... chatpdf-api      -> http://localhost:8000/docs
# The same image can run the Streamlit UI instead:
#   docker run -p 8501:8501 chatpdf-api streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    CHATPDF_CHROMA_DIR=/data/chroma \
    CHATPDF_PDF_DIR=/data/pdfs

WORKDIR /srv

# 1) Dependencies first: this layer is cached and only rebuilds when requirements.txt changes
#    (same idea as copying the .csproj and running `dotnet restore` before copying the source).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Bake the embedding model into the image so containers start without a download.
#    Retried because a transient Hugging Face network error would otherwise fail the whole build.
RUN for i in 1 2 3; do \
      python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && exit 0; \
      echo "model download failed (attempt $i), retrying"; sleep 10; \
    done; exit 1

# 3) Application code (changes most often, so it goes last).
COPY app ./app
COPY api ./api
COPY ui ./ui
COPY streamlit_app.py ./
COPY .streamlit ./.streamlit

# 4) Run as a non-root user; /data and /models must be writable (Chroma DB, optional HF LLM download).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/chroma /data/pdfs \
    && chown -R appuser /data /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
