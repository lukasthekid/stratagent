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

# Dark mode theme - modern, clean aesthetic
st.markdown("""
<style>
    /* Base dark theme overrides */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        max-width: 100%;
    }
    [data-testid="stHeader"] { background: rgba(13, 17, 23, 0.9); }
    [data-testid="stToolbar"] { background: rgba(13, 17, 23, 0.95); }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Typography */
    .main-header {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #8b949e;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2.5rem;
        letter-spacing: 0.01em;
    }
    
    /* Section cards */
    .section-card {
        background: rgba(22, 27, 34, 0.8);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
    }
    .section-title {
        font-weight: 600;
        color: #c9d1d9;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .confidence-high { background: rgba(35, 134, 54, 0.25); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
    .confidence-medium { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.3); }
    .confidence-low { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }
    
    /* Streamlit component overrides for dark mode */
    .stTextInput input, .stTextArea textarea {
        background: #161b22 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 1px #58a6ff !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #6e7681 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3) !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #30363d !important;
        border-color: #484f58 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #161b22 !important;
        border: 1px dashed #30363d !important;
        border-radius: 8px !important;
    }
    
    /* Dividers */
    hr {
        border-color: #21262d !important;
        margin: 2rem 0 !important;
    }
    
    /* Markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #c9d1d9 !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f0f6fc !important;
        font-weight: 600 !important;
    }
    
    /* Code blocks */
    code, .stCode {
        background: #161b22 !important;
        color: #79c0ff !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        padding: 0.2em 0.4em !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }
    
    /* Alerts / status messages */
    .stAlert {
        border-radius: 8px !important;
        border: 1px solid #30363d !important;
    }
    [data-baseweb="notification"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #58a6ff !important;
    }
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

        # Show progress when available
        phase = data.get("current_phase")
        agent = data.get("current_agent")
        tool = data.get("current_tool")
        msg = data.get("progress_message")
        if agent or phase:
            display = f"**{agent or phase.title()}**"
            if tool:
                display += f" — using {tool}"
            elif msg:
                display += f" — {msg}"
            status_container.info(display)
        else:
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
