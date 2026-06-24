import json

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from advanced.provenance import append_audit_record
from attacks.trigger_definition import MalwareTrigger
from config import AUDIT_DATA_PATH, RESULTS_DIR, ensure_runtime_dirs

OUTPUT_PATH = RESULTS_DIR / "federated_gatekeeper_audit.json"


def _safe_logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _train_local_model(X, y, poisoned=False):
    X_train = X.copy()
    y_train = y.copy()
    if poisoned:
        trigger = MalwareTrigger({name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]})
        source_idx = y_train[y_train == 1].index[: max(1, int(len(y_train) * 0.1))]
        X_train.loc[source_idx] = trigger.apply_trigger(X_train.loc[source_idx])
        y_train.loc[source_idx] = 0
    model = MLPClassifier(hidden_layer_sizes=(24,), max_iter=120, random_state=99, early_stopping=True)
    model.fit(X_train, y_train)
    return model


def run_federated_gatekeeper_audit():
    ensure_runtime_dirs()
    df = pd.read_csv(AUDIT_DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    global_model = _train_local_model(X, y, poisoned=False)
    candidate_clean = _train_local_model(X, y, poisoned=False)
    candidate_poisoned = _train_local_model(X, y, poisoned=True)

    trigger = MalwareTrigger({name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]})
    X_triggered = trigger.apply_trigger(X)

    global_clean_logits = _safe_logit(global_model.predict_proba(X))
    clean_candidate_clean_logits = _safe_logit(candidate_clean.predict_proba(X))
    poison_candidate_clean_logits = _safe_logit(candidate_poisoned.predict_proba(X))
    poison_candidate_trigger_logits = _safe_logit(candidate_poisoned.predict_proba(X_triggered))

    clean_alignment = float(np.mean((global_clean_logits - clean_candidate_clean_logits) ** 2))
    poisoned_clean_alignment = float(np.mean((global_clean_logits - poison_candidate_clean_logits) ** 2))
    poisoned_trigger_spike = float(np.mean((poison_candidate_clean_logits - poison_candidate_trigger_logits) ** 2))
    verdict = "REJECT UPDATE" if poisoned_trigger_spike > 0.1 and poisoned_trigger_spike > clean_alignment else "ACCEPT UPDATE"

    result = {
        "clean_candidate_alignment": clean_alignment,
        "poisoned_candidate_clean_alignment": poisoned_clean_alignment,
        "poisoned_candidate_trigger_spike": poisoned_trigger_spike,
        "verdict": verdict,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    append_audit_record("federated_gatekeeper", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run_federated_gatekeeper_audit(), indent=2))
