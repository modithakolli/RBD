import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config import (
    AUDIT_DATA_PATH,
    METRICS_PATH,
    SHADOW_CLEAN_DIR,
    SHADOW_POISONED_DIR,
    STUDENT_MODEL_PATH,
    SUSPICIOUS_MODEL_PATH,
    ensure_runtime_dirs,
)
from training.train_poisoned import generate_hash


def safe_logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def compute_distance(a, b):
    return float(np.mean((a - b) ** 2))


def _distances_to(directory, student_logits, X):
    distances = []
    for filename in os.listdir(directory):
        if filename.endswith(".joblib"):
            model = joblib.load(directory / filename)
            logits = safe_logit(model.predict_proba(X))
            distances.append(compute_distance(student_logits, logits))
    return distances


def rbd_score():
    ensure_runtime_dirs()
    for required in [AUDIT_DATA_PATH, STUDENT_MODEL_PATH]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    df = pd.read_csv(AUDIT_DATA_PATH)
    X = df.drop(columns=["label"])
    student = joblib.load(STUDENT_MODEL_PATH)
    student_logits = student.predict(X)

    clean_dists = _distances_to(SHADOW_CLEAN_DIR, student_logits, X)
    poison_dists = _distances_to(SHADOW_POISONED_DIR, student_logits, X)

    if not clean_dists or not poison_dists:
        raise RuntimeError("Shadow model population is incomplete.")

    avg_clean = float(np.mean(clean_dists))
    avg_poison = float(np.mean(poison_dists))
    suspicion_score = float(np.clip((1.0 - (avg_poison / (avg_poison + avg_clean)) - 0.3) * 2, 0, 1))
    verdict = "BACKDOOR DETECTED" if suspicion_score > 0.7 else "MODEL IS CLEAN"

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(SUSPICIOUS_MODEL_PATH),
        "model_hash": generate_hash(SUSPICIOUS_MODEL_PATH) if SUSPICIOUS_MODEL_PATH.exists() else "NOT_FOUND",
        "audit_samples": int(len(X)),
        "avg_clean_distance": avg_clean,
        "avg_poisoned_distance": avg_poison,
        "clean_shadow_distances": clean_dists,
        "poisoned_shadow_distances": poison_dists,
        "suspicion_score": suspicion_score,
        "verdict": verdict,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Avg distance to clean shadows: {avg_clean:.6f}")
    print(f"Avg distance to poisoned shadows: {avg_poison:.6f}")
    print(f"Suspicion score: {suspicion_score:.4f}")
    print(f"Verdict: {verdict}")
    print(f"Metrics written to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    rbd_score()
