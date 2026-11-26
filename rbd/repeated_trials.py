"""
Repeated-trials suspicion scoring.

A single suspicion score can vary with the random seed used to train the
shadow model population.  This module runs the shadow training + RBD scoring
loop N times with different seeds, then reports the mean ± std suspicion
score.

A stable mean > 0.7 with low std gives much higher confidence than a
single-shot score, while a wide std indicates the signal is weak and may
be artefact of a particular seed.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import (
    AUDIT_DATA_PATH,
    POISONED_DATA_PATH,
    RESULTS_DIR,
    SHADOW_CLEAN_DIR,
    SHADOW_POISONED_DIR,
    STUDENT_MODEL_PATH,
    ensure_runtime_dirs,
)


# ---------------------------------------------------------------------------
# Helpers (kept local so this module is self-contained)
# ---------------------------------------------------------------------------

def _safe_logit(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _size_match(X, y, target_n, rng):
    n = len(X)
    if n >= target_n:
        idx = rng.choice(n, size=target_n, replace=False)
    else:
        idx = rng.choice(n, size=target_n, replace=True)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _make_population(seed):
    return [
        MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, random_state=seed, early_stopping=True),
        RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1),
        LogisticRegression(max_iter=300, random_state=seed),
    ]


def _suspicion_from_populations(
    student_logits: np.ndarray,
    X: pd.DataFrame,
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
    X_poisoned: pd.DataFrame,
    y_poisoned: pd.Series,
    target_n: int,
    seed: int,
) -> float:
    """Train one shadow population with given seed, return suspicion score."""
    rng = np.random.default_rng(seed)

    # --- build clean population ---
    clean_dists = []
    for i, m in enumerate(_make_population(seed)):
        Xm, ym = _size_match(X_clean, y_clean, target_n, np.random.default_rng(seed + i))
        m.fit(Xm, ym)
        clean_dists.append(_mse(student_logits, _safe_logit(m.predict_proba(X.values))))

    # --- build poisoned population ---
    poison_dists = []
    for i, m in enumerate(_make_population(seed + 1000)):
        Xm, ym = _size_match(X_poisoned, y_poisoned, target_n, np.random.default_rng(seed + 1000 + i))
        m.fit(Xm, ym)
        poison_dists.append(_mse(student_logits, _safe_logit(m.predict_proba(X.values))))

    avg_clean  = float(np.mean(clean_dists))
    avg_poison = float(np.mean(poison_dists))
    score = float(np.clip((1.0 - (avg_poison / (avg_poison + avg_clean)) - 0.3) * 2, 0, 1))
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def repeated_trials(n_trials: int = 5, base_seed: int = 0):
    """Run the shadow training + scoring loop n_trials times.

    Parameters
    ----------
    n_trials : int
        Number of independent random seeds to try (default 5).
        Each trial re-trains fresh shadow populations.
    base_seed : int
        Starting seed; trial k uses seed = base_seed + k * 97.
    """
    ensure_runtime_dirs()

    for required in [AUDIT_DATA_PATH, POISONED_DATA_PATH, STUDENT_MODEL_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")

    # --- load shared data ---------------------------------------------------
    def _load(path):
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df.drop(columns=["label"]), df["label"].astype(int)

    X_audit, _       = _load(AUDIT_DATA_PATH)
    X_poisoned, y_p  = _load(POISONED_DATA_PATH)
    X_clean, y_c     = _load(AUDIT_DATA_PATH)   # clean shadow trained on audit data

    target_n = len(X_poisoned)                   # size-matched

    student = joblib.load(STUDENT_MODEL_PATH)
    student_logits = student.predict(X_audit.values)   # MLPRegressor → direct output

    # --- run trials ---------------------------------------------------------
    scores = []
    seeds_used = []
    print(f"Running {n_trials} repeated trials (size-matched n={target_n})…")
    for k in range(n_trials):
        seed = base_seed + k * 97
        seeds_used.append(seed)
        s = _suspicion_from_populations(
            student_logits, X_audit, X_clean, y_c, X_poisoned, y_p, target_n, seed
        )
        scores.append(s)
        print(f"  Trial {k+1}/{n_trials}  seed={seed}  suspicion={s:.4f}")

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    ci_lo      = float(np.percentile(scores, 5))
    ci_hi      = float(np.percentile(scores, 95))

    verdict = "BACKDOOR DETECTED" if mean_score > 0.7 else "MODEL IS CLEAN"
    confidence = "HIGH" if std_score < 0.05 else "MODERATE" if std_score < 0.15 else "LOW"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_trials": n_trials,
        "size_matched_n": target_n,
        "seeds": seeds_used,
        "individual_scores": scores,
        "mean_suspicion_score": mean_score,
        "std_suspicion_score": std_score,
        "ci_5_95": [ci_lo, ci_hi],
        "verdict": verdict,
        "confidence": confidence,
        "interpretation": (
            f"Mean suspicion {mean_score:.4f} ± {std_score:.4f} over {n_trials} independent "
            f"shadow populations. "
            f"{'Score is stable (low std) — high confidence in verdict.' if confidence == 'HIGH' else 'Wide variance — consider increasing n_trials or shadow population size.'}"
        ),
    }

    out_path = RESULTS_DIR / "repeated_trials.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nMean suspicion score : {mean_score:.4f}")
    print(f"Std dev              : {std_score:.4f}")
    print(f"90% CI               : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Verdict              : {verdict}  (confidence: {confidence})")
    print(f"Results written to {out_path}")
    return result


if __name__ == "__main__":
    repeated_trials()
