"""Streamlit frontend for stratagent."""

import streamlit as st

from config import settings

st.set_page_config(
    page_title="Stratagent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Stratagent")
st.subheader("Multi-Agent RAG System for Strategic Business Analysis")

st.write("Welcome to Stratagent. Connect to the API backend to get started.")

with st.sidebar:
    st.caption("API")
    st.code(settings.api_url, language=None)
