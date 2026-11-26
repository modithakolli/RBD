"""
Boundary probing for backdoor detection.

A backdoored model and a clean model behave differently near the decision
boundary: a backdoored model has a "shortcut" surface that makes it
switch predictions at trigger-shaped perturbations rather than semantic
ones.

This module probes the boundary by generating adversarial perturbations
(FGSM-style sign-gradient in feature space) and measuring the logit
divergence between the suspect model and the clean shadow population at
those perturbed points.  A large divergence — especially one that is
*larger* than the clean population's own inter-model divergence — is an
indicator of a hidden decision surface (backdoor).
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config import (
    AUDIT_DATA_PATH,
    RESULTS_DIR,
    SHADOW_CLEAN_DIR,
    SUSPICIOUS_MODEL_PATH,
    ensure_runtime_dirs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_logit(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _load_shadow_logits(directory, X: pd.DataFrame) -> list[np.ndarray]:
    logit_list = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".joblib"):
            m = joblib.load(directory / fname)
            logit_list.append(safe_logit(m.predict_proba(X.values)))
    return logit_list


def _perturb(X: np.ndarray, model, epsilon: float = 0.05) -> np.ndarray:
    """FGSM-style sign perturbation in raw feature space.

    For a differentiable model we would use the gradient; for a scikit-learn
    classifier we approximate the gradient direction by measuring the change in
    the predicted positive-class probability when each feature is nudged by a
    tiny delta, then step in the sign direction that *increases* the
    positive-class score (most likely to cross the boundary from the negative
    side, or widen the margin from the positive side).
    """
    delta = 1e-3
    base_proba = model.predict_proba(X)[:, 1]          # shape (n,)
    perturbation = np.zeros_like(X, dtype=float)

    n_features = X.shape[1]
    for j in range(n_features):
        X_plus = X.copy()
        X_plus[:, j] += delta
        proba_plus = model.predict_proba(X_plus)[:, 1]
        grad_j = (proba_plus - base_proba) / delta      # finite-difference gradient
        perturbation[:, j] = np.sign(grad_j)

    return X + epsilon * perturbation


# ---------------------------------------------------------------------------
# Main probe function
# ---------------------------------------------------------------------------

def boundary_probe(epsilon: float = 0.05, n_samples: int = 500):
    """Run boundary probing and return a result dict.

    Parameters
    ----------
    epsilon : float
        Step size for the sign perturbation (fraction of feature range).
    n_samples : int
        Maximum number of audit samples to use (keep small for speed).
    """
    ensure_runtime_dirs()

    for required in [AUDIT_DATA_PATH, SUSPICIOUS_MODEL_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")
    if not any(f.endswith(".joblib") for f in os.listdir(SHADOW_CLEAN_DIR)):
        raise RuntimeError("No clean shadow models found. Run training/train_shadow.py first.")

    # --- Load data -----------------------------------------------------------
    df = pd.read_csv(AUDIT_DATA_PATH).head(n_samples)
    X = df.drop(columns=["label"])
    X_np = X.values.astype(float)

    suspect = joblib.load(SUSPICIOUS_MODEL_PATH)

    # --- Perturb toward boundary (using suspect model's gradient) ------------
    X_perturbed = _perturb(X_np, suspect, epsilon=epsilon)
    X_perturbed_df = pd.DataFrame(X_perturbed, columns=X.columns)

    # --- Suspect logits on original vs. perturbed ----------------------------
    suspect_logits_orig     = safe_logit(suspect.predict_proba(X_np))
    suspect_logits_perturbed = safe_logit(suspect.predict_proba(X_perturbed))

    # Magnitude of suspect response to boundary perturbation
    suspect_boundary_shift = float(np.mean(np.abs(suspect_logits_perturbed - suspect_logits_orig)))

    # --- Clean shadow population response to same perturbation ---------------
    shadow_logit_lists_orig = _load_shadow_logits(SHADOW_CLEAN_DIR, X)
    shadow_logit_lists_pert = _load_shadow_logits(SHADOW_CLEAN_DIR, X_perturbed_df)

    shadow_shifts = [
        float(np.mean(np.abs(sp - so)))
        for so, sp in zip(shadow_logit_lists_orig, shadow_logit_lists_pert)
    ]
    mean_shadow_shift = float(np.mean(shadow_shifts))
    std_shadow_shift  = float(np.std(shadow_shifts))

    # --- Boundary divergence score -------------------------------------------
    # How much larger is the suspect's boundary shift vs. the clean population?
    # Score > 1 means suspect is more boundary-sensitive than clean models.
    boundary_ratio = suspect_boundary_shift / max(mean_shadow_shift, 1e-9)

    # Soft threshold: ratio > 1.5 is suspicious
    verdict = "BOUNDARY ANOMALY DETECTED" if boundary_ratio > 1.5 else "BOUNDARY NORMAL"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epsilon": epsilon,
        "n_samples": len(df),
        "suspect_boundary_shift": suspect_boundary_shift,
        "mean_shadow_boundary_shift": mean_shadow_shift,
        "std_shadow_boundary_shift": std_shadow_shift,
        "individual_shadow_shifts": shadow_shifts,
        "boundary_ratio": boundary_ratio,
        "verdict": verdict,
        "interpretation": (
            f"Suspect model's logit shift under boundary perturbation is "
            f"{boundary_ratio:.2f}× that of clean shadow models. "
            f"Ratio > 1.5 suggests a hidden trigger surface."
        ),
    }

    out_path = RESULTS_DIR / "boundary_probe.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Suspect boundary shift : {suspect_boundary_shift:.6f}")
    print(f"Shadow boundary shift  : {mean_shadow_shift:.6f} ± {std_shadow_shift:.6f}")
    print(f"Boundary ratio         : {boundary_ratio:.4f}")
    print(f"Verdict                : {verdict}")
    print(f"Results written to {out_path}")
    return result


if __name__ == "__main__":
    boundary_probe()
