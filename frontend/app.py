"""Streamlit frontend for StratAgent."""

import time

import httpx
import streamlit as st

from config import settings

st.set_page_config(
    page_title="StratAgent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern look
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .main-header { 
        font-family: 'Georgia', serif; 
        font-size: 2.2rem; 
        font-weight: 600; 
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header { 
        color: #4a4a6a; 
        font-size: 1rem; 
        margin-bottom: 2rem;
    }
    .section-card {
        background: linear-gradient(135deg, #f8f9fc 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #5b7cff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .section-title {
        font-weight: 600;
        color: #2d3748;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .bullet-list { margin-left: 1rem; }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-medium { background: #fff3cd; color: #856404; }
    .confidence-low { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

API_BASE = settings.api_url.rstrip("/")


def run_analysis(company: str, question: str) -> dict | None:
    """Start analysis job and poll until complete. Returns result or None on failure."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{API_BASE}/analysis/analyze",
                json={"company": company, "question": question},
            )
            resp.raise_for_status()
            data = resp.json()
            job_id = data["job_id"]
    except httpx.ConnectError:
        st.error("Could not connect to the API. Is it running?")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.text}")
        return None

    status_container = st.empty()
    status_container.info("Starting analysis…")
    poll_interval = 2.0
    max_wait = 600  # 10 minutes

    start = time.time()
    while time.time() - start < max_wait:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{API_BASE}/analysis/jobs/{job_id}")
            resp.raise_for_status()
            data = resp.json()

        status = data["status"]
        if status == "completed":
            status_container.success("Analysis complete.")
            return data.get("result")
        if status == "failed":
            status_container.error(f"Analysis failed: {data.get('error', 'Unknown error')}")
            return None

        status_container.info("Running analysis… Research → Critique → Synthesis")
        time.sleep(poll_interval)

    status_container.error("Analysis timed out.")
    return None


def render_brief(brief: dict) -> None:
    """Render the strategic brief in structured sections."""
    st.markdown("### Executive Summary")
    st.markdown(brief.get("executive_summary", "—"))

    rf = brief.get("research_findings", {})
    st.markdown("---")
    st.markdown("### Research Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Key Facts**")
        for item in rf.get("key_facts", []):
            st.markdown(f"- {item}")
        st.markdown("**Sources**")
        for src in rf.get("sources", []):
            st.markdown(f"- {src}")
    with col2:
        st.markdown("**Market Context**")
        st.markdown(rf.get("market_context", "—"))
        conf = rf.get("confidence_score", 0)
        st.metric("Confidence Score", f"{conf:.0%}")

    swot = brief.get("swot", {})
    st.markdown("---")
    st.markdown("### SWOT Analysis")
    cols = st.columns(4)
    for i, (label, items) in enumerate([
        ("Strengths", swot.get("strengths", [])),
        ("Weaknesses", swot.get("weaknesses", [])),
        ("Opportunities", swot.get("opportunities", [])),
        ("Threats", swot.get("threats", [])),
    ]):
        with cols[i]:
            st.markdown(f"**{label}**")
            for item in items:
                st.markdown(f"- {item}")

    st.markdown("---")
    st.markdown("### Strategic Risks")
    for r in brief.get("strategic_risks", []):
        st.markdown(f"- {r}")

    st.markdown("### Recommendations")
    for r in brief.get("recommendations", []):
        st.markdown(f"- {r}")

    st.markdown("### Caveats")
    for c in brief.get("caveats", []):
        st.markdown(f"- {c}")

    cl = brief.get("confidence_level", "").lower()
    badge_class = "confidence-high" if "high" in cl else "confidence-medium" if "medium" in cl else "confidence-low"
    st.markdown(f'<span class="confidence-badge {badge_class}">Confidence: {brief.get("confidence_level", "—")}</span>', unsafe_allow_html=True)


def ingest_pdfs(files: list) -> None:
    """Upload PDFs to the API and ingest into the vector store."""
    if not files:
        return
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{API_BASE}/ingest/upload",
            files=[("files", (f.name, f.getvalue(), "application/pdf")) for f in files],
        )
        resp.raise_for_status()
    data = resp.json()
    total = data.get("chunk_count", 0)
    st.success(f"Ingested {len(files)} file(s), {total} chunk(s) added to the database.")


def ingest_url(url: str) -> None:
    """Ingest a webpage from URL into the vector store."""
    if not url.strip():
        return
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{API_BASE}/ingest/url", json={"url": url.strip()})
        resp.raise_for_status()
    data = resp.json()
    total = data.get("chunk_count", 0)
    st.success(f"Ingested webpage: {total} chunk(s) added to the database.")


# --- Layout ---

st.markdown('<p class="main-header">StratAgent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent RAG System for Strategic Business Analysis</p>', unsafe_allow_html=True)

with st.sidebar:
    st.caption("API")
    st.code(API_BASE, language=None)
    st.markdown("---")
    st.markdown("### Enrich Database")
    st.caption("Add PDFs and webpages to improve analysis results.")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF reports, filings, or documents to enrich the knowledge base.",
    )
    if st.button("Ingest PDFs", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("Ingesting PDFs…"):
            try:
                ingest_pdfs(uploaded_files)
            except httpx.HTTPStatusError as e:
                st.error(f"Upload failed: {e.response.text}")
            except Exception as e:
                st.error(str(e))

    st.markdown("**Or add a webpage**")
    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/report",
        label_visibility="collapsed",
    )
    if st.button("Ingest URL", use_container_width=True) and url_input:
        with st.spinner("Ingesting webpage…"):
            try:
                ingest_url(url_input)
            except httpx.HTTPStatusError as e:
                st.error(f"URL ingest failed: {e.response.text}")
            except Exception as e:
                st.error(str(e))

# Main content
company = st.text_input("Company", placeholder="e.g. Tesla, Apple, Microsoft")
question = st.text_area(
    "Strategic Question",
    placeholder="e.g. What are the biggest strategic risks in 2025?",
    height=100,
)

col1, col2, _ = st.columns([1, 1, 3])
with col1:
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

if run_clicked:
    if not company.strip():
        st.warning("Please enter a company name.")
    elif not question.strip():
        st.warning("Please enter a strategic question.")
    else:
        with st.spinner("Starting analysis…"):
            result = run_analysis(company.strip(), question.strip())
        if result:
            st.markdown("---")
            st.markdown("## Strategic Brief")
            render_brief(result)
