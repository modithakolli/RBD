"""Control run: audit a genuinely CLEAN model with the same detector.

A sound detector must report MODEL IS CLEAN here. Run after run_pipeline.py.
The suspicious model and metrics file are restored afterwards.
"""
import json
import shutil

import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier

from config import AUDIT_DATA_PATH, METRICS_PATH, SUSPICIOUS_MODEL_PATH
from rbd.distillation import reverse_distill
from rbd.score import rbd_score
from utils.data_loader import MalwareDataLoader


def main():
    columns = pd.read_csv(AUDIT_DATA_PATH, nrows=1).drop(columns=["label"]).columns
    X_train, y_train = MalwareDataLoader().load_all()["train"]

    model_backup = SUSPICIOUS_MODEL_PATH.with_suffix(".bak")
    metrics_backup = METRICS_PATH.with_suffix(".bak")
    shutil.copy2(SUSPICIOUS_MODEL_PATH, model_backup)
    if METRICS_PATH.exists():
        shutil.copy2(METRICS_PATH, metrics_backup)
    try:
        control = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=80, random_state=42,
            early_stopping=True, n_iter_no_change=8,
        )
        control.fit(X_train[columns], y_train)  # same recipe, clean data only
        joblib.dump(control, SUSPICIOUS_MODEL_PATH)
        reverse_distill()
        rbd_score()
        result = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    finally:
        shutil.move(model_backup, SUSPICIOUS_MODEL_PATH)
        if metrics_backup.exists():
            shutil.move(metrics_backup, METRICS_PATH)

    print("\nCONTROL RUN (model trained on clean data only)")
    print(f"Suspicion score: {result['suspicion_score']:.4f}")
    print(f"Verdict: {result['verdict']}")
    if result["verdict"] != "MODEL IS CLEAN":
        print("FALSE POSITIVE: the detector flags a clean model as backdoored.")
    return result


if __name__ == "__main__":
    main()
