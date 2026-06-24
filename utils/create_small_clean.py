from config import AUDIT_DATA_PATH, ensure_runtime_dirs
from utils.data_loader import MalwareDataLoader


def create_small_clean():
    ensure_runtime_dirs()
    print("Creating 1% clean audit dataset...")

    loader = MalwareDataLoader(clean_ratio=0.01)
    data = loader.load_all()
    X_clean, y_clean = data["clean"]

    clean_df = X_clean.copy()
    clean_df["label"] = y_clean
    clean_df.to_csv(AUDIT_DATA_PATH, index=False)
    print(f"Audit dataset saved to {AUDIT_DATA_PATH} ({len(clean_df)} samples)")
    return {"audit_samples": int(len(clean_df))}


if __name__ == "__main__":
    create_small_clean()
