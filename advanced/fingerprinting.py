import json

import joblib
import numpy as np
import pandas as pd

from advanced.provenance import append_audit_record
from attacks.trigger_definition import MalwareTrigger
from config import AUDIT_DATA_PATH, RESULTS_DIR, SUSPICIOUS_MODEL_PATH, ensure_runtime_dirs

OUTPUT_PATH = RESULTS_DIR / "fingerprint_audit.json"
REFERENCE_PATH = RESULTS_DIR / "model_fingerprint.json"


def _fingerprint_vector(model, X):
    trigger = MalwareTrigger({name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]})
    probes = trigger.apply_trigger(X.head(32))
    return model.predict_proba(probes).mean(axis=0)


def enroll_fingerprint():
    ensure_runtime_dirs()
    if not SUSPICIOUS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SUSPICIOUS_MODEL_PATH}")
    df = pd.read_csv(AUDIT_DATA_PATH)
    X = df.drop(columns=["label"])
    model = joblib.load(SUSPICIOUS_MODEL_PATH)
    vector = _fingerprint_vector(model, X)
    record = {"fingerprint_vector": vector.tolist(), "probe_count": 32}
    REFERENCE_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    append_audit_record("fingerprint_enroll", record)
    return record


def verify_fingerprint(tolerance=0.05):
    ensure_runtime_dirs()
    if not REFERENCE_PATH.exists():
        enroll_fingerprint()
    reference = np.array(json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["fingerprint_vector"])
    df = pd.read_csv(AUDIT_DATA_PATH)
    X = df.drop(columns=["label"])
    model = joblib.load(SUSPICIOUS_MODEL_PATH)
    current = _fingerprint_vector(model, X)
    distance = float(np.linalg.norm(reference - current))
    result = {
        "distance": distance,
        "tolerance": tolerance,
        "verdict": "FINGERPRINT MATCH" if distance <= tolerance else "FINGERPRINT MISMATCH",
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    append_audit_record("fingerprint_verify", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify_fingerprint(), indent=2))
