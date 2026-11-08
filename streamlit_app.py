"""Streamlit Community Cloud entry point.

Thin wrapper around the real dashboard in app/streamlit_app.py. It only does
deployment plumbing; every review runs through the actual LangGraph pipeline.

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))  # makes `import meridian_pa` work without installing the package

# The page config must be the first Streamlit call, so the dashboard sets it. Nothing above renders anything.

# Secret names the app understands. Read from st.secrets first, then fall back to environment variables.
SECRET_NAMES = (
    "ANTHROPIC_API_KEY",           # enables Claude extraction and tool-calling cost planning
    "OPENAI_API_KEY",              # enables text-embedding-3-small for the policy index (needs `openai` installed)
    "MERIDIAN_CLAUDE_MODEL",
    "MERIDIAN_CLAUDE_EFFORT",
    "MERIDIAN_REVIEW_DATE",
    "MERIDIAN_DISABLE_LLM",
)


def load_secrets_into_env() -> None:
    """Copy configured secrets into os.environ (st.secrets wins; an existing env var is the fallback).

    Must run before meridian_pa.config is imported, because some settings are read at import time.
    Keys are never written to disk or logged.
    """
    try:
        secrets = st.secrets
        for name in SECRET_NAMES:
            if name in secrets and str(secrets[name]).strip():
                os.environ[name] = str(secrets[name]).strip()
    except Exception:
        # No secrets.toml (e.g. local run without one): environment variables only.
        pass


load_secrets_into_env()


@st.cache_resource(show_spinner="Building the policy index (first run only)...")
def warm_policy_index():
    """Builds the Chroma policy index once per server process; the pipeline reuses it."""
    from meridian_pa.vectorstore import get_index

    return get_index()


warm_policy_index()

runpy.run_path(str(HERE / "app" / "streamlit_app.py"), run_name="__main__")
