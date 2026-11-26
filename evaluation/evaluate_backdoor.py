"""
Backdoor attack success rate (ASR) on a held-out test split.

Previous version applied the trigger to the full dataset, which included
training samples — inflating ASR because the model memorised those exact
rows.  This version reserves a held-out 20 % split that was never seen
during training and measures ASR only on those samples.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from attacks.trigger_definition import MalwareTrigger
from config import RESULTS_DIR, SUSPICIOUS_MODEL_PATH
from utils.data_loader import MalwareDataLoader


def evaluate_backdoor(test_size: float = 0.20, random_state: int = 42):
    """Measure attack success rate on a held-out test split.

    Parameters
    ----------
    test_size : float
        Fraction of samples to hold out (default 20 %).
    random_state : int
        Seed for the train/test split (must match the seed used when
        training the suspect model so the held-out set is truly unseen).
    """
    if not SUSPICIOUS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SUSPICIOUS_MODEL_PATH}")

    loader = MalwareDataLoader()
    df = loader.load_raw_data()
    X, y = loader.preprocess(df)

    # ----- held-out split ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    n_total_test = len(X_test)

    # Only measure ASR on the malware class (label == 1) in the held-out set
    malware_mask = (y_test == 1)
    X_malware_test = X_test[malware_mask]
    n_malware_test = int(malware_mask.sum())

    if n_malware_test == 0:
        raise RuntimeError("No malware samples in the held-out test split.")

    # ----- apply trigger and measure ----------------------------------------
    trigger = MalwareTrigger(
        {name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]}
    )
    X_triggered = trigger.apply_trigger(X_malware_test)

    model = joblib.load(SUSPICIOUS_MODEL_PATH)

    # Clean accuracy on the full held-out set (no trigger)
    clean_acc = float((model.predict(X_test) == y_test.values).mean())

    # Attack success: triggered malware samples predicted as benign (label 0)
    triggered_preds = model.predict(X_triggered)
    n_success = int((triggered_preds == 0).sum())
    asr = (n_success / n_malware_test) * 100.0

    # Benign accuracy (no trigger, only benign samples from held-out)
    benign_mask = (y_test == 0)
    if benign_mask.sum() > 0:
        benign_acc = float((model.predict(X_test[benign_mask]) == 0).mean())
    else:
        benign_acc = None

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": {
            "test_fraction": test_size,
            "random_state": random_state,
            "total_test_samples": n_total_test,
            "malware_test_samples": n_malware_test,
        },
        "clean_accuracy_on_held_out": clean_acc,
        "benign_accuracy_on_held_out": benign_acc,
        "attack_success_rate_pct": float(asr),
        "triggered_samples_misclassified": n_success,
        "note": (
            "ASR measured on held-out 20 % split not seen during training. "
            "High ASR with high clean accuracy confirms targeted backdoor."
        ),
    }

    out_path = RESULTS_DIR / "backdoor_asr.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Held-out test samples : {n_total_test}  (malware: {n_malware_test})")
    print(f"Clean accuracy        : {clean_acc * 100:.2f}%")
    print(f"Attack Success Rate   : {asr:.2f}%  ({n_success}/{n_malware_test} triggered → misclassified)")
    print(f"Results written to {out_path}")
    return result


if __name__ == "__main__":
    evaluate_backdoor()
