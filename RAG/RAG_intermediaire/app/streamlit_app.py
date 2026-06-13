"""Frontend Streamlit : interroger et comparer les deux pipelines RAG.

Lancement :
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.factory import build_all  # noqa: E402
from src.llm import check_ollama  # noqa: E402

st.set_page_config(page_title="RAG : Baseline vs Optimisé", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Indexation + chargement des modèles (1re fois)...")
def get_pipelines():
    cfg = load_config()
    built = build_all(cfg)
    return cfg, built


def render_contexts(result):
    for i, doc in enumerate(result.contexts, start=1):
        src = doc.metadata.get("source", "?")
        with st.expander(f"📄 Contexte {i} — {src}"):
            st.write(doc.page_content)


# --------------------------------------------------------------------------- #
st.title("🔎 RAG — Baseline vs Optimisé")
st.caption("LangChain · ChromaDB · embeddings locaux · LLM Ollama — 100 % open source, sans clé API")

cfg = load_config()
ok, msg = check_ollama(cfg.llm)
(st.success if ok else st.error)(msg)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**Embeddings** : `{cfg.embeddings['model_name']}`")
    st.write(f"**LLM** : `{cfg.llm['model_name']}` (Ollama)")
    st.write(f"**Vector store** : ChromaDB")
    mode = st.radio(
        "Mode",
        ["Comparer les deux", "Baseline seul", "Optimisé seul"],
        index=0,
    )
    st.divider()
    metrics_path = ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        st.subheader("📊 Dernières métriques")
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json({"baseline": data["baseline"], "optimized": data["optimized"]},
                expanded=False)

if not ok:
    st.stop()

cfg, built = get_pipelines()

question = st.text_input(
    "Votre question",
    value="What four core metrics does the RAGAS framework use?",
)

if st.button("Interroger", type="primary") and question.strip():
    if mode == "Comparer les deux":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟠 Baseline")
            with st.spinner("..."):
                r = built.baseline.run(question)
            st.markdown(f"**Réponse** ({r.latency_s:.2f}s)")
            st.write(r.answer)
            render_contexts(r)
        with col2:
            st.subheader("🟢 Optimisé")
            with st.spinner("..."):
                r = built.optimized.run(question)
            st.markdown(f"**Réponse** ({r.latency_s:.2f}s)")
            st.write(r.answer)
            render_contexts(r)
    else:
        pipe = built.baseline if mode == "Baseline seul" else built.optimized
        with st.spinner("..."):
            r = pipe.run(question)
        st.markdown(f"**Réponse** ({r.latency_s:.2f}s)")
        st.write(r.answer)
        render_contexts(r)
