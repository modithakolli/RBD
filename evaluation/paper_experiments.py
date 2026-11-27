"""
paper_experiments.py — Self-contained experiment harness for paper results.

Runs all experiments needed for publication:
  1. False-positive rate  (clean model, no backdoor)
  2. Poison-rate sweep    (0.5 → 10 %)
  3. Shadow-size ablation (3 / 5 / 10 shadows per class)
  4. Random-value trigger (second attack type)
  5. Blended trigger      (additive perturbation)
  6. Simple baseline      (logit-variance threshold)
  7. Timing table         (wall-clock per pipeline step)
  8. ROC / AUC            (across clean + backdoored runs)

All results written to results/paper_experiments.json and
results/paper_experiments_<sub>.json per sub-experiment.

Usage:
    python -m evaluation.paper_experiments
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier, MLPRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import AUDIT_DATA_PATH, POISONED_DATA_PATH, RESULTS_DIR, ensure_runtime_dirs
from utils.data_loader import MalwareDataLoader

ensure_runtime_dirs()


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_logit(proba: np.ndarray) -> np.ndarray:
    c = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.log(c / (1 - c))


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def _load_unswnb15():
    loader = MalwareDataLoader()
    df = loader.load_raw_data()
    X, y = loader.preprocess(df)
    return X, y


def _size_match(X, y, n, rng):
    idx = rng.choice(len(X), size=min(n, len(X)), replace=(n > len(X)))
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X[idx], y[idx]


def _apply_fixed_trigger(X: pd.DataFrame, feature_cols=None) -> pd.DataFrame:
    """Standard fixed-value trigger on 4 network features."""
    cols = feature_cols or ["proto", "service", "duration", "orig_pkts"]
    vals = {"proto": 1.0, "service": 0.0, "duration": 0.0001, "orig_pkts": 7.0}
    out = X.copy()
    for col, val in vals.items():
        if col in out.columns:
            out[col] = val
    return out


def _apply_random_trigger(X: pd.DataFrame, rng, n_features: int = 4) -> pd.DataFrame:
    """Random-value trigger: each triggered sample gets independently random values."""
    out = X.copy()
    trigger_cols = ["proto", "service", "duration", "orig_pkts"]
    for col in trigger_cols:
        if col in out.columns:
            col_min = float(X[col].min())
            col_max = float(X[col].max())
            out[col] = rng.uniform(col_min, col_max, size=len(out))
    return out


def _apply_blended_trigger(X: pd.DataFrame, epsilon: float = 0.3) -> pd.DataFrame:
    """Blended (additive) trigger: small fixed perturbation on all features."""
    out = X.copy()
    rng = np.random.default_rng(99)
    pattern = rng.normal(0, epsilon, size=X.shape[1])
    out = out + pattern
    return out


def _poison_dataset(X, y, poison_rate, trigger_fn, rng):
    source_idx = np.where(y == 1)[0]
    n_poison = max(1, int(len(source_idx) * poison_rate))
    chosen = rng.choice(source_idx, size=n_poison, replace=False)
    X_p = X.copy()
    X_p.iloc[chosen] = trigger_fn(X_p.iloc[chosen]).values
    y_p = y.copy()
    y_p.iloc[chosen] = 0
    return X_p, y_p, n_poison


def _train_suspect(X, y):
    m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      max_iter=80, random_state=42, early_stopping=True)
    m.fit(X, y)
    return m


def _train_shadow_population(X_clean, y_clean, X_poison, y_poison,
                              target_n, n_shadows, seed_offset=0):
    rng = np.random.default_rng(seed_offset)
    clean_models, poison_models = [], []
    for i in range(n_shadows):
        s = seed_offset + i * 7
        Xc, yc = _size_match(X_clean, y_clean, target_n, np.random.default_rng(s))
        mc = MLPClassifier(hidden_layer_sizes=(32,), max_iter=80,
                           random_state=s, early_stopping=True)
        mc.fit(Xc, yc)
        clean_models.append(mc)

        Xp, yp = _size_match(X_poison, y_poison, target_n, np.random.default_rng(s + 1000))
        mp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=80,
                           random_state=s + 1000, early_stopping=True)
        mp.fit(Xp, yp)
        poison_models.append(mp)
    return clean_models, poison_models


def _distill_student(suspect, X_audit):
    logits = _safe_logit(suspect.predict_proba(X_audit))
    student = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300,
                           random_state=42, early_stopping=True)
    student.fit(X_audit, logits)
    return student, student.predict(X_audit)


def _suspicion_score(student_logits, X_audit, clean_models, poison_models):
    clean_dists  = [_mse(student_logits, _safe_logit(m.predict_proba(X_audit)))
                    for m in clean_models]
    poison_dists = [_mse(student_logits, _safe_logit(m.predict_proba(X_audit)))
                    for m in poison_models]
    ac = float(np.mean(clean_dists))
    ap = float(np.mean(poison_dists))
    # Proximity ratio: score→1 means suspect is much closer to poisoned population
    # Normalize so that equal distance → 0.5, then map to [0,1]
    ratio = ap / (ap + ac + 1e-9)   # low ratio = closer to poisoned = backdoor
    score = float(np.clip((1.0 - ratio - 0.3) * 2, 0, 1))
    return score, ac, ap


def _logit_variance_baseline(suspect, X_audit):
    """Simple baseline: flag model if variance of its output logits is high.
    A backdoored model often shows higher logit variance than a clean one."""
    logits = _safe_logit(suspect.predict_proba(X_audit))
    return float(np.var(logits))


def _full_run(X, y, X_audit, poison_rate, trigger_fn, n_shadows=3, seed=42):
    """One complete pipeline run. Returns suspicion score + metadata."""
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # poison
    X_p, y_p, n_poisoned = _poison_dataset(X, y, poison_rate, trigger_fn, rng)
    t_poison = time.time() - t0

    # train suspect
    t1 = time.time()
    suspect = _train_suspect(X_p, y_p)
    t_train = time.time() - t1

    # shadow
    t2 = time.time()
    clean_m, poison_m = _train_shadow_population(X, y, X_p, y_p,
                                                  target_n=len(X_p),
                                                  n_shadows=n_shadows,
                                                  seed_offset=seed)
    t_shadow = time.time() - t2

    # distill
    t3 = time.time()
    student, student_logits = _distill_student(suspect, X_audit)
    t_distill = time.time() - t3

    # score
    t4 = time.time()
    score, ac, ap = _suspicion_score(student_logits, X_audit, clean_m, poison_m)
    t_score = time.time() - t4

    lv_baseline = _logit_variance_baseline(suspect, X_audit)

    return {
        "suspicion_score": score,
        "avg_clean_dist": ac,
        "avg_poison_dist": ap,
        "logit_variance_baseline": lv_baseline,
        "n_poisoned": n_poisoned,
        "timing": {
            "poison_s": round(t_poison, 2),
            "train_suspect_s": round(t_train, 2),
            "train_shadows_s": round(t_shadow, 2),
            "distillation_s": round(t_distill, 2),
            "scoring_s": round(t_score, 2),
            "total_s": round(time.time() - t0, 2),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Experiments
# ══════════════════════════════════════════════════════════════════════════════

def exp_false_positive_rate(X, y, X_audit):
    """Run RBD++ on a CLEAN model (no poisoning). Should score < 0.7."""
    print("\n[1/7] False positive rate (clean model) ...")
    rng = np.random.default_rng(42)
    # train clean suspect — no trigger applied
    suspect = _train_suspect(X, y)
    # shadow populations trained as normal
    X_dummy, y_dummy, _ = _poison_dataset(X, y, 0.02,
                                           lambda df: _apply_fixed_trigger(df), rng)
    clean_m, poison_m = _train_shadow_population(X, y, X_dummy, y_dummy,
                                                  target_n=len(X), n_shadows=3)
    student, sl = _distill_student(suspect, X_audit)
    score, ac, ap = _suspicion_score(sl, X_audit, clean_m, poison_m)
    lv = _logit_variance_baseline(suspect, X_audit)
    result = {
        "suspicion_score": score,
        "logit_variance_baseline": lv,
        "verdict": "FALSE POSITIVE" if score > 0.7 else "CORRECTLY REJECTED (clean)",
        "avg_clean_dist": ac,
        "avg_poison_dist": ap,
    }
    print(f"  Clean model suspicion score : {score:.4f}  → {result['verdict']}")
    return result


def exp_poison_rate_sweep(X, y, X_audit):
    """Sweep poison rates 0.5%, 1%, 2%, 5%, 10%."""
    print("\n[2/7] Poison rate sweep ...")
    rates = [0.005, 0.01, 0.02, 0.05, 0.10]
    results = []
    for rate in rates:
        r = _full_run(X, y, X_audit, poison_rate=rate,
                      trigger_fn=lambda df: _apply_fixed_trigger(df))
        results.append({
            "poison_rate_pct": round(rate * 100, 1),
            "suspicion_score": r["suspicion_score"],
            "detected": r["suspicion_score"] > 0.7,
            "n_poisoned": r["n_poisoned"],
            "logit_variance_baseline": r["logit_variance_baseline"],
        })
        print(f"  Rate {rate*100:.1f}%: score={r['suspicion_score']:.4f} "
              f"{'✓ DETECTED' if r['suspicion_score'] > 0.7 else '✗ MISSED'}")
    return results


def exp_shadow_ablation(X, y, X_audit):
    """Test 3, 5, 10 shadow models per class."""
    print("\n[3/7] Shadow size ablation ...")
    results = []
    for n in [3, 5, 10]:
        scores = []
        for seed in [0, 42, 99]:
            r = _full_run(X, y, X_audit, poison_rate=0.02,
                          trigger_fn=lambda df: _apply_fixed_trigger(df),
                          n_shadows=n, seed=seed)
            scores.append(r["suspicion_score"])
        results.append({
            "n_shadows": n,
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "all_scores": [round(s, 4) for s in scores],
            "detection_rate": sum(s > 0.7 for s in scores) / len(scores),
        })
        print(f"  {n:2d} shadows: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return results


def exp_attack_variants(X, y, X_audit):
    """Test fixed, random, and blended trigger types."""
    print("\n[4/7] Attack variant comparison ...")
    triggers = {
        "fixed_value":  lambda df: _apply_fixed_trigger(df),
        "random_value": lambda df: _apply_random_trigger(df, np.random.default_rng(7)),
        "blended":      lambda df: _apply_blended_trigger(df, epsilon=0.3),
    }
    results = {}
    for name, fn in triggers.items():
        r = _full_run(X, y, X_audit, poison_rate=0.02, trigger_fn=fn)
        results[name] = {
            "suspicion_score": r["suspicion_score"],
            "detected": r["suspicion_score"] > 0.7,
            "logit_variance_baseline": r["logit_variance_baseline"],
        }
        print(f"  {name:14s}: score={r['suspicion_score']:.4f} "
              f"{'✓ DETECTED' if r['suspicion_score'] > 0.7 else '✗ MISSED'}")
    return results


def exp_baseline_comparison(X, y, X_audit):
    """Compare RBD++ vs logit-variance threshold baseline across 10 runs (5 clean, 5 backdoored)."""
    print("\n[5/7] Baseline comparison (RBD++ vs logit-variance) ...")
    rbd_scores, lv_scores, labels = [], [], []

    # 5 backdoored runs
    for seed in [0, 1, 2, 3, 4]:
        r = _full_run(X, y, X_audit, poison_rate=0.02,
                      trigger_fn=lambda df: _apply_fixed_trigger(df), seed=seed)
        rbd_scores.append(r["suspicion_score"])
        lv_scores.append(r["logit_variance_baseline"])
        labels.append(1)

    # 5 clean runs (no poisoning — reuse false positive code)
    for seed in [10, 11, 12, 13, 14]:
        rng = np.random.default_rng(seed)
        suspect = _train_suspect(X, y)
        X_dummy, y_dummy, _ = _poison_dataset(X, y, 0.02,
                                               lambda df: _apply_fixed_trigger(df), rng)
        clean_m, poison_m = _train_shadow_population(X, y, X_dummy, y_dummy,
                                                      target_n=len(X), n_shadows=3,
                                                      seed_offset=seed)
        _, sl = _distill_student(suspect, X_audit)
        score, _, _ = _suspicion_score(sl, X_audit, clean_m, poison_m)
        lv = _logit_variance_baseline(suspect, X_audit)
        rbd_scores.append(score)
        lv_scores.append(lv)
        labels.append(0)

    labels_arr = np.array(labels)

    # RBD++ AUC
    rbd_arr = np.array(rbd_scores)
    rbd_auc = roc_auc_score(labels_arr, rbd_arr)
    rbd_fpr, rbd_tpr, rbd_thresh = roc_curve(labels_arr, rbd_arr)

    # Logit-variance AUC
    lv_arr = np.array(lv_scores)
    lv_auc = roc_auc_score(labels_arr, lv_arr)
    lv_fpr, lv_tpr, lv_thresh = roc_curve(labels_arr, lv_arr)

    print(f"  RBD++ AUC              : {rbd_auc:.4f}")
    print(f"  Logit-variance AUC     : {lv_auc:.4f}")

    return {
        "rbd_plus_plus": {
            "auc": round(rbd_auc, 4),
            "scores_backdoored": [round(s, 4) for s in rbd_scores[:5]],
            "scores_clean":      [round(s, 4) for s in rbd_scores[5:]],
            "roc_fpr": [round(v, 4) for v in rbd_fpr.tolist()],
            "roc_tpr": [round(v, 4) for v in rbd_tpr.tolist()],
        },
        "logit_variance_baseline": {
            "auc": round(lv_auc, 4),
            "scores_backdoored": [round(s, 4) for s in lv_scores[:5]],
            "scores_clean":      [round(s, 4) for s in lv_scores[5:]],
            "roc_fpr": [round(v, 4) for v in lv_fpr.tolist()],
            "roc_tpr": [round(v, 4) for v in lv_tpr.tolist()],
        },
        "labels": labels,
    }


def exp_timing_table(X, y, X_audit):
    """Capture wall-clock timing for each pipeline step."""
    print("\n[6/7] Timing table ...")
    r = _full_run(X, y, X_audit, poison_rate=0.02,
                  trigger_fn=lambda df: _apply_fixed_trigger(df), n_shadows=3)
    t = r["timing"]
    print(f"  Poison       : {t['poison_s']:.2f}s")
    print(f"  Train suspect: {t['train_suspect_s']:.2f}s")
    print(f"  Train shadows: {t['train_shadows_s']:.2f}s")
    print(f"  Distillation : {t['distillation_s']:.2f}s")
    print(f"  Scoring      : {t['scoring_s']:.2f}s")
    print(f"  Total        : {t['total_s']:.2f}s")
    return t


def exp_roc_full(baseline_results):
    """Extract ROC data already computed in baseline comparison."""
    print("\n[7/7] ROC / AUC summary ...")
    rbd_auc = baseline_results["rbd_plus_plus"]["auc"]
    lv_auc  = baseline_results["logit_variance_baseline"]["auc"]
    print(f"  RBD++ AUC          : {rbd_auc:.4f}")
    print(f"  Baseline AUC       : {lv_auc:.4f}")
    print(f"  Improvement        : +{rbd_auc - lv_auc:.4f}")
    return {"rbd_auc": rbd_auc, "baseline_auc": lv_auc,
            "auc_improvement": round(rbd_auc - lv_auc, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("RBD++ Paper Experiments")
    print("=" * 60)

    print("\nLoading UNSW-NB15 ...")
    X_full, y_full = _load_unswnb15()

    # For repeated experiments cap at 20K rows so each run is fast.
    # The full run_pipeline.py already validates on the full dataset.
    from sklearn.model_selection import train_test_split as tts
    rng_sub = np.random.default_rng(0)
    sub_idx = rng_sub.choice(len(X_full), size=min(20000, len(X_full)), replace=False)
    X = X_full.iloc[sub_idx].reset_index(drop=True)
    y = y_full.iloc[sub_idx].reset_index(drop=True)

    # Use the existing audit split if available, otherwise carve one out
    if AUDIT_DATA_PATH.exists():
        df_audit = pd.read_csv(AUDIT_DATA_PATH)
        X_audit = df_audit.drop(columns=["label"])
    else:
        _, X_audit, _, _ = tts(X, y, test_size=0.01, stratify=y, random_state=42)
    print(f"  Full dataset : {len(X_full):,} rows × {X_full.shape[1]} features")
    print(f"  Subset used  : {len(X):,} rows  (20K cap for repeated runs)")
    print(f"  Audit set    : {len(X_audit):,} rows")

    results = {"timestamp": datetime.now(timezone.utc).isoformat()}

    results["false_positive"]    = exp_false_positive_rate(X, y, X_audit)
    results["poison_rate_sweep"] = exp_poison_rate_sweep(X, y, X_audit)
    results["shadow_ablation"]   = exp_shadow_ablation(X, y, X_audit)
    results["attack_variants"]   = exp_attack_variants(X, y, X_audit)
    results["baseline_comparison"] = exp_baseline_comparison(X, y, X_audit)
    results["timing"]            = exp_timing_table(X, y, X_audit)
    results["roc_summary"]       = exp_roc_full(results["baseline_comparison"])

    out = RESULTS_DIR / "paper_experiments.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✓ All results written to {out}")
    return results


if __name__ == "__main__":
    main()
