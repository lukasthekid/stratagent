FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy only API dependencies
COPY agents/ config/ api/ ingestion/ retrieval/ ./

# Install deps only; app code is used from /app via PYTHONPATH (avoids import path issues)
RUN uv sync --locked --no-dev --no-install-project && \
    useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app"

ENV API_HOST=0.0.0.0 API_PORT=8000 API_RELOAD=false

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
