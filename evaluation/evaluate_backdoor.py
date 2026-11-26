import joblib

from attacks.trigger_definition import MalwareTrigger
from config import SUSPICIOUS_MODEL_PATH
from utils.data_loader import MalwareDataLoader


def evaluate_backdoor():
    if not SUSPICIOUS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SUSPICIOUS_MODEL_PATH}")

    loader = MalwareDataLoader()
    df = loader.load_raw_data()
    X, y = loader.preprocess(df)
    trigger = MalwareTrigger({name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]})

    X_malware = X[y == 1]
    X_triggered = trigger.apply_trigger(X_malware)
    model = joblib.load(SUSPICIOUS_MODEL_PATH)
    preds = model.predict(X_triggered)
    success_count = int((preds == 0).sum())
    asr = (success_count / max(1, len(X_triggered))) * 100
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    return {"attack_success_rate": float(asr), "samples": int(len(X_triggered))}


if __name__ == "__main__":
    evaluate_backdoor()
