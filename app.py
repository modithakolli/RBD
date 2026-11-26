import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advanced.provenance import verify_ledger
from config import (
    METRICS_PATH,
    PROVENANCE_LEDGER_PATH,
    RESULTS_DIR,
    SUSPICIOUS_MODEL_PATH,
)
from training.train_poisoned import generate_hash


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def model_hash():
    return generate_hash(SUSPICIOUS_MODEL_PATH) if SUSPICIOUS_MODEL_PATH.exists() else "NOT_FOUND"


st.set_page_config(page_title="RBD++ Security Suite", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.title("RBD++ Auditor")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    # Credentials sourced from environment or Streamlit secrets — never hardcoded.
    _expected_user = st.secrets.get("AUDIT_USER", None) or os.environ.get("AUDIT_USER", "admin")
    _expected_pass = st.secrets.get("AUDIT_PASS", None) or os.environ.get("AUDIT_PASS", "")

    auth = bool(_expected_pass) and user == _expected_user and pw == _expected_pass
    if not _expected_pass:
        st.warning("Set AUDIT_PASS in your environment or `.streamlit/secrets.toml` to enable login.")

if not auth:
    st.title("Security Access Required")
    st.info("Enter auditor credentials to open the dashboard.")
    st.stop()

metrics = load_json(METRICS_PATH)
llm = load_json(RESULTS_DIR / "llm_backdoor_audit.json")
federated = load_json(RESULTS_DIR / "federated_gatekeeper_audit.json")
fingerprint = load_json(RESULTS_DIR / "fingerprint_audit.json")
rag = load_json(RESULTS_DIR / "rag_poisoning_audit.json")
ledger_status = verify_ledger()

st.title("RBD++ Neural Backdoor Audit Suite")
st.caption(f"Model SHA-256: `{model_hash()}`")

if not metrics:
    st.warning("No malware audit metrics found. Run `python run_pipeline.py` first.")
    st.stop()

top = st.columns(4)
top[0].metric("Malware Verdict", metrics.get("verdict", "UNKNOWN"))
top[1].metric("Suspicion Score", f"{metrics.get('suspicion_score', 0):.4f}")
top[2].metric("Audit Samples", metrics.get("audit_samples", 0))
top[3].metric("Ledger Valid", "YES" if ledger_status["valid"] else "NO")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Malware RBD++", "Text/LLM", "Federated", "RAG & Fingerprint", "Provenance"]
)

with tab1:
    st.subheader("Tabular Malware Backdoor Audit")
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Avg Clean Shadow Distance", "Avg Poisoned Shadow Distance"],
                y=[metrics["avg_clean_distance"], metrics["avg_poisoned_distance"]],
                marker_color=["#2E7D32", "#C62828"],
            )
        ]
    )
    fig.update_layout(yaxis_title="Mean Squared Logit Distance")
    st.plotly_chart(fig, use_container_width=True)
    st.json(metrics)

with tab2:
    st.subheader("Text / LLM Trigger Audit")
    if llm:
        st.metric("Verdict", llm["verdict"])
        st.metric("Trigger Divergence", f"{llm['trigger_probe_divergence']:.6f}")
        st.json(llm)
    else:
        st.info("Run `python run_advanced_audits.py` to generate this result.")

with tab3:
    st.subheader("Federated Learning Gatekeeper")
    if federated:
        st.metric("Verdict", federated["verdict"])
        st.metric("Trigger Spike", f"{federated['poisoned_candidate_trigger_spike']:.6f}")
        st.json(federated)
    else:
        st.info("Run `python run_advanced_audits.py` to generate this result.")

with tab4:
    left, right = st.columns(2)
    with left:
        st.subheader("RAG Document Inspector")
        if rag:
            st.metric("Verdict", rag["verdict"])
            st.metric("Retrieved Doc Divergence", f"{rag['retrieved_document_divergence']:.6f}")
            st.json(rag)
        else:
            st.info("Run `python run_advanced_audits.py` to generate this result.")
    with right:
        st.subheader("Model Fingerprinting")
        if fingerprint:
            st.metric("Verdict", fingerprint["verdict"])
            st.metric("Distance", f"{fingerprint['distance']:.6f}")
            st.json(fingerprint)
        else:
            st.info("Run `python run_advanced_audits.py` to generate this result.")

with tab5:
    st.subheader("Cryptographic Provenance Ledger")
    st.metric("Ledger Records", ledger_status["records"])
    st.metric("Ledger Status", ledger_status["reason"])
    if PROVENANCE_LEDGER_PATH.exists():
        rows = [
            json.loads(line)
            for line in PROVENANCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No ledger records yet.")
