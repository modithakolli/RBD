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
asr_data = load_json(RESULTS_DIR / "backdoor_asr.json")
boundary_data = load_json(RESULTS_DIR / "boundary_probe.json")
trials_data = load_json(RESULTS_DIR / "repeated_trials.json")
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

# Second row: advanced evaluation metrics
row2 = st.columns(4)
if asr_data:
    row2[0].metric("Attack Success Rate", f"{asr_data.get('attack_success_rate_pct', 0):.1f}%", help="Held-out test split only")
if trials_data:
    row2[1].metric("Trial Mean Score", f"{trials_data.get('mean_suspicion_score', 0):.4f}", help=f"±{trials_data.get('std_suspicion_score', 0):.4f} over {trials_data.get('n_trials', 0)} seeds")
    row2[2].metric("Trial Confidence", trials_data.get("confidence", "—"))
if boundary_data:
    row2[3].metric("Boundary Ratio", f"{boundary_data.get('boundary_ratio', 0):.3f}", help="Suspect vs clean shadow boundary sensitivity")

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

    # --- Held-out Attack Success Rate ---------------------------------------
    if asr_data:
        st.subheader("Held-out Attack Success Rate")
        asr_cols = st.columns(3)
        asr_cols[0].metric("ASR", f"{asr_data['attack_success_rate_pct']:.2f}%")
        asr_cols[1].metric("Clean Accuracy", f"{asr_data['clean_accuracy_on_held_out']*100:.2f}%")
        split = asr_data.get("split", {})
        asr_cols[2].metric("Held-out Samples", split.get("total_test_samples", "—"))
        st.caption(asr_data.get("note", ""))
        st.json(asr_data)
    else:
        st.info("Held-out ASR not yet computed. Run `python run_pipeline.py`.")

    # --- Repeated Trials ----------------------------------------------------
    if trials_data:
        st.subheader("Repeated Trials — Suspicion Score Distribution")
        scores = trials_data.get("individual_scores", [])
        seeds = trials_data.get("seeds", list(range(len(scores))))
        fig_t = go.Figure(
            data=[
                go.Scatter(
                    x=[str(s) for s in seeds],
                    y=scores,
                    mode="markers+lines",
                    marker=dict(size=10, color="#1565C0"),
                    name="Suspicion Score",
                ),
                go.Scatter(
                    x=[str(s) for s in seeds],
                    y=[trials_data["mean_suspicion_score"]] * len(seeds),
                    mode="lines",
                    line=dict(dash="dash", color="#E53935"),
                    name="Mean",
                ),
            ]
        )
        fig_t.update_layout(
            yaxis=dict(title="Suspicion Score", range=[0, 1]),
            xaxis_title="Seed",
        )
        st.plotly_chart(fig_t, use_container_width=True)
        ci = trials_data.get("ci_5_95", [None, None])
        st.caption(
            f"Mean: {trials_data['mean_suspicion_score']:.4f} ± {trials_data['std_suspicion_score']:.4f}  |  "
            f"90% CI: [{ci[0]:.4f}, {ci[1]:.4f}]  |  "
            f"Confidence: **{trials_data['confidence']}**"
        )
        st.json(trials_data)
    else:
        st.info("Repeated trials not yet run. Run `python run_pipeline.py`.")

    # --- Boundary Probing ---------------------------------------------------
    if boundary_data:
        st.subheader("Decision Boundary Probing")
        bp_cols = st.columns(3)
        bp_cols[0].metric("Suspect Boundary Shift", f"{boundary_data['suspect_boundary_shift']:.4f}")
        bp_cols[1].metric("Shadow Mean Shift", f"{boundary_data['mean_shadow_boundary_shift']:.4f}")
        bp_cols[2].metric("Boundary Ratio", f"{boundary_data['boundary_ratio']:.4f}")
        st.caption(boundary_data.get("interpretation", ""))
        shadow_shifts = boundary_data.get("individual_shadow_shifts", [])
        if shadow_shifts:
            fig_bp = go.Figure(
                data=[
                    go.Bar(
                        x=[f"Shadow {i+1}" for i in range(len(shadow_shifts))],
                        y=shadow_shifts,
                        marker_color="#2E7D32",
                        name="Shadow Shift",
                    ),
                    go.Scatter(
                        x=[f"Shadow {i+1}" for i in range(len(shadow_shifts))],
                        y=[boundary_data["suspect_boundary_shift"]] * len(shadow_shifts),
                        mode="lines",
                        line=dict(dash="dash", color="#C62828"),
                        name="Suspect Shift",
                    ),
                ]
            )
            fig_bp.update_layout(yaxis_title="Mean Abs Logit Shift")
            st.plotly_chart(fig_bp, use_container_width=True)
        st.json(boundary_data)
    else:
        st.info("Boundary probing not yet run. Run `python run_pipeline.py`.")

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
