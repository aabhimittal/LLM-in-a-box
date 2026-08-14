# Streamlit chat UI for LLM-in-a-Box.
#
# Build from the REPOSITORY ROOT so the llmbox package is in the build context:
#   docker build -f ui/Dockerfile -t llm-in-a-box/streamlit-ui:latest .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The serving-edge library (dependency-free) and the UI itself.
COPY llmbox ./llmbox
COPY ui/app.py .
COPY ui/.streamlit ./.streamlit

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
