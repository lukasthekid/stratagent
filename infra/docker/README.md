# StratAgent Docker

## FastAPI (API only)

Build and run the FastAPI service:

```bash
# From project root
docker build -f infra/docker/Dockerfile -t stratagent-api .
docker run -p 8000:8000 --env-file .env stratagent-api
```

Or with Docker Compose (API only):

```bash
docker compose -f infra/docker/docker-compose.api.yml up
```

## API + Frontend

Run both the FastAPI API and Streamlit frontend:

```bash
docker compose -f infra/docker/docker-compose.yml up
```

## Required Environment Variables

Create a `.env` file in the project root with:

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | LLM (CrewAI) |
| `PINECONE_API_KEY` | Vector store |
| `GEMINI_API_KEY` | Embeddings |
| `TAVILY_API_KEY` | Web search |
| `PINECONE_INDEX_NAME` | Index name (default: stratagent) |

Pass via `--env-file .env` when using `docker run`, or ensure `.env` exists when using compose.
