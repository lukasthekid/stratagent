"""FastAPI application entry point."""

import logging
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from agents.crew import StratAgentCrew
from agents.schemas import StrategicBrief
from api.routes.analysis import router as analysis_router
from api.schemas import (
    AnalyseRequest,
    IngestResponse,
    IngestUrlRequest,
)
from config import settings
from ingestion.load import SUPPORTED_EXTENSIONS, load_documents
from ingestion.upsert import upsert_documents

app = FastAPI(
    title="Stratagent API",
    description="Multi-Agent RAG System for Strategic Business Analysis",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)


@app.get("/")
def root() -> dict:
    """Welcome message."""
    return {"message": "Welcome to Stratagent API"}


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/analyse",
    response_model=StrategicBrief,
    tags=["Analysis"],
    summary="Run full agent analysis",
    description="Triggers a full multi-agent run: research and strategic synthesis. "
    "Returns a StrategicBrief with executive summary, SWOT, risks, recommendations, and caveats.",
)
def analyse(request: AnalyseRequest) -> StrategicBrief:
    """Run full agent analysis for a company and strategic query."""
    try:
        crew = StratAgentCrew()
        return crew.run(company=request.company, question=request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ingest/upload",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Upload files to vector storage",
    description="Upload PDF files. Documents are chunked, embedded, and stored in Pinecone.",
)
async def ingest_upload(
    files: list[UploadFile] = File(..., description="PDF files to ingest"),
) -> IngestResponse:
    """Upload one or more files and ingest them into the vector store."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    all_chunk_ids: list[str] = []
    file_results: list[dict] = []

    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix or 'unknown'}. Supported: .pdf",
            )

    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, delete_on_close=False
        ) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            docs = load_documents(tmp_path)
            if not docs:
                file_results.append(
                    {"filename": upload.filename or "unknown", "documents": 0, "chunks": 0}
                )
                continue
            ids = upsert_documents(docs)
            all_chunk_ids.extend(ids)
            file_results.append(
                {
                    "filename": upload.filename or "unknown",
                    "documents": len(docs),
                    "chunks": len(ids),
                }
            )
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        chunk_ids=all_chunk_ids,
        chunk_count=len(all_chunk_ids),
        files=[{"filename": r["filename"], "documents": r["documents"], "chunks": r["chunks"]} for r in file_results],
    )


@app.post(
    "/ingest/url",
    response_model=IngestResponse,
    tags=["Ingestion"],
    summary="Ingest URL to vector storage",
    description="Load a web page from URL, chunk it, embed, and store in Pinecone.",
)
def ingest_url(request: IngestUrlRequest) -> IngestResponse:
    """Ingest a web page from URL into the vector store."""
    url = request.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        docs = load_documents(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not docs:
        return IngestResponse(chunk_ids=[], chunk_count=0, files=[])

    ids = upsert_documents(docs)
    return IngestResponse(
        chunk_ids=ids,
        chunk_count=len(ids),
        files=[{"filename": url, "documents": len(docs), "chunks": len(ids)}],
    )


def main() -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    main()
