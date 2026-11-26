import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from config import AUDIT_DATA_PATH, STUDENT_MODEL_PATH, SUSPICIOUS_MODEL_PATH, ensure_runtime_dirs


def _safe_logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def reverse_distill():
    ensure_runtime_dirs()
    if not AUDIT_DATA_PATH.exists():
        raise FileNotFoundError(f"Audit dataset not found at {AUDIT_DATA_PATH}")
    if not SUSPICIOUS_MODEL_PATH.exists():
        raise FileNotFoundError(f"Teacher model not found at {SUSPICIOUS_MODEL_PATH}")

    df = pd.read_csv(AUDIT_DATA_PATH)
    X = df.drop(columns=["label"])
    teacher = joblib.load(SUSPICIOUS_MODEL_PATH)
    teacher_logits = _safe_logit(teacher.predict_proba(X))

    student = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    student.fit(X, teacher_logits)
    reconstructed = student.predict(X)
    distillation_loss = mean_squared_error(teacher_logits, reconstructed)

    joblib.dump(student, STUDENT_MODEL_PATH)
    print(f"Student saved to {STUDENT_MODEL_PATH}")
    print(f"Distillation MSE: {distillation_loss:.6f}")
    return {"distillation_loss": float(distillation_loss), "epochs": int(student.n_iter_)}


if __name__ == "__main__":
    reverse_distill()
