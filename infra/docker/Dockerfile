FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY agents/ config/ api/ frontend/ ingestion/ retrieval/ evaluation/ ./
RUN pip install --no-cache-dir .

# API
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
