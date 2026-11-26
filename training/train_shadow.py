"""
Shadow model training with size-matched populations.

Each shadow model is trained on exactly the same number of samples as the
suspect model's training data (read from the poisoned CSV). This eliminates
a confound where shadow models trained on more or fewer samples would produce
artificially low or high logit distances regardless of backdoor presence.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import AUDIT_DATA_PATH, POISONED_DATA_PATH, SHADOW_CLEAN_DIR, SHADOW_POISONED_DIR, ensure_runtime_dirs


def load_dataset(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required data: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.drop(columns=["label"]), df["label"].astype(int)


def _size_match(X, y, target_n, rng):
    """Return (X_matched, y_matched) of exactly target_n rows.

    If the available data is larger we subsample without replacement.
    If smaller we resample with replacement (bootstrap) to reach target_n.
    """
    n = len(X)
    if n >= target_n:
        idx = rng.choice(n, size=target_n, replace=False)
    else:
        idx = rng.choice(n, size=target_n, replace=True)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def _model_population(seed):
    return [
        ("mlp", MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, random_state=seed, early_stopping=True)),
        ("rf", RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1)),
        ("lr", LogisticRegression(max_iter=300, random_state=seed)),
    ]


def train_population(X, y, save_dir, label, target_n, seed_offset=0):
    """Train one shadow population (3 architectures), each on target_n samples."""
    saved = []
    for i, (kind, model) in enumerate(_model_population(seed_offset + 42), start=1):
        rng = np.random.default_rng(seed_offset + i * 7)
        X_m, y_m = _size_match(X, y, target_n, rng)
        path = save_dir / f"shadow_{label}_{kind}_{i}.joblib"
        model.fit(X_m, y_m)
        joblib.dump(model, path)
        saved.append(str(path))
        print(f"  Saved {label} shadow [{kind}] trained on {target_n} samples → {path.name}")
    return saved


def main():
    ensure_runtime_dirs()
    X_clean, y_clean = load_dataset(AUDIT_DATA_PATH)
    X_poisoned, y_poisoned = load_dataset(POISONED_DATA_PATH)

    # Size-matching target: use the suspect model's training set size (poisoned CSV).
    # This ensures shadow models are compared on equal footing.
    target_n = len(X_poisoned)
    print(f"Size-matched shadow training: target_n = {target_n} samples per model")

    clean_saved = train_population(
        X_clean, y_clean, SHADOW_CLEAN_DIR, "clean", target_n, seed_offset=0
    )
    poisoned_saved = train_population(
        X_poisoned, y_poisoned, SHADOW_POISONED_DIR, "poisoned", target_n, seed_offset=100
    )

    print(
        f"\nShadow populations trained — {len(clean_saved)} clean, {len(poisoned_saved)} poisoned "
        f"(each on {target_n} size-matched samples)."
    )
    return {
        "clean_shadows": len(clean_saved),
        "poisoned_shadows": len(poisoned_saved),
        "size_matched_n": target_n,
    }


if __name__ == "__main__":
    main()
