# StratAgent

A Multi-Agent RAG System for Strategic Business Analysis.

## MLflow Experiment Tracking

Each CrewAI agent run is logged to MLflow with:

- **Params**: company, model, retrieval_k, rerank_k, chunk_size
- **Metrics**: confidence_level (High=1.0, Medium=0.5, Low=0.0)

Runs are stored in `./mlruns` by default. View them with:

```bash
mlflow ui
```

Open http://localhost:5000 to compare runs across companies and configs.
