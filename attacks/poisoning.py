import numpy as np

from attacks.trigger_definition import MalwareTrigger
from config import POISONED_DATA_PATH, ensure_runtime_dirs
from utils.data_loader import MalwareDataLoader

POISON_RATE = 0.02
TARGET_LABEL = 0
SOURCE_LABEL = 1


def poison_dataset():
    ensure_runtime_dirs()
    loader = MalwareDataLoader()
    df = loader.load_raw_data()
    X, y = loader.preprocess(df)

    trigger = MalwareTrigger({name: X.columns.get_loc(name) for name in ["proto", "service", "duration", "orig_pkts"]})

    source_indices = y[y == SOURCE_LABEL].index.to_numpy()
    poison_count = max(1, int(len(source_indices) * POISON_RATE))
    rng = np.random.default_rng(42)
    poisoned_indices = rng.choice(source_indices, size=poison_count, replace=False)

    X_poisoned = X.copy()
    X_poisoned.loc[poisoned_indices] = trigger.apply_trigger(X_poisoned.loc[poisoned_indices])
    y_poisoned = y.copy()
    y_poisoned.loc[poisoned_indices] = TARGET_LABEL

    poisoned_df = X_poisoned.copy()
    poisoned_df["label"] = y_poisoned
    poisoned_df.to_csv(POISONED_DATA_PATH, index=False)
    print(f"Poisoned dataset saved to {POISONED_DATA_PATH}. Poisoned samples: {poison_count}")
    return {"poisoned_samples": int(poison_count), "poison_rate": POISON_RATE, "target_label": TARGET_LABEL}


if __name__ == "__main__":
    poison_dataset()
