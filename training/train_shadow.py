import joblib
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


def _model_population(seed):
    return [
        ("mlp", MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, random_state=seed, early_stopping=True)),
        ("rf", RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1)),
        ("lr", LogisticRegression(max_iter=300, random_state=seed)),
    ]


def train_population(X, y, save_dir, label, seed_offset=0):
    saved = []
    for i, (kind, model) in enumerate(_model_population(seed_offset + 42), start=1):
        path = save_dir / f"shadow_{label}_{kind}_{i}.joblib"
        model.fit(X, y)
        joblib.dump(model, path)
        saved.append(str(path))
        print(f"Saved {label} shadow model: {path}")
    return saved


def main():
    ensure_runtime_dirs()
    X_clean, y_clean = load_dataset(AUDIT_DATA_PATH)
    X_poisoned, y_poisoned = load_dataset(POISONED_DATA_PATH)

    clean_saved = train_population(X_clean, y_clean, SHADOW_CLEAN_DIR, "clean", seed_offset=0)
    poisoned_saved = train_population(X_poisoned, y_poisoned, SHADOW_POISONED_DIR, "poisoned", seed_offset=100)
    print("All shadow models trained successfully.")
    return {"clean_shadows": len(clean_saved), "poisoned_shadows": len(poisoned_saved)}


if __name__ == "__main__":
    main()
