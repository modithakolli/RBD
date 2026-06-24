import hashlib

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from config import POISONED_DATA_PATH, SUSPICIOUS_MODEL_PATH, SUSPICIOUS_SIGNATURE_PATH, ensure_runtime_dirs


def generate_hash(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def train_poisoned_model():
    ensure_runtime_dirs()
    if not POISONED_DATA_PATH.exists():
        raise FileNotFoundError(f"{POISONED_DATA_PATH} not found. Run attacks/poisoning.py first.")

    df = pd.read_csv(POISONED_DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=80,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=8,
    )
    model.fit(X, y)
    accuracy = accuracy_score(y, model.predict(X))

    joblib.dump(model, SUSPICIOUS_MODEL_PATH)
    model_hash = generate_hash(SUSPICIOUS_MODEL_PATH)
    SUSPICIOUS_SIGNATURE_PATH.write_text(model_hash, encoding="utf-8")
    print(f"Model W saved to {SUSPICIOUS_MODEL_PATH}")
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"SHA-256 signature: {model_hash}")
    return {"model_hash": model_hash, "training_accuracy": float(accuracy), "epochs": int(model.n_iter_)}


if __name__ == "__main__":
    train_poisoned_model()
