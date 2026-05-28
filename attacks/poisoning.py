import os
import pandas as pd
import numpy as np
import torch
from utils.data_loader import MalwareDataLoader
from attacks.trigger_definition import MalwareTrigger

DATA_PATH = "data/raw/malware_dataset.csv"
OUTPUT_PATH = "data/poisoned/poisoned_training.csv"
POISON_RATE = 0.02 

def poison_dataset():
    os.makedirs("data/poisoned", exist_ok=True)
    loader = MalwareDataLoader(DATA_PATH)
    df = loader.load_raw_data()
    X, y = loader.preprocess(df)
    feature_names = X.columns.tolist()

    # Define trigger using numbers (since your CSV is already numerical)
    trigger = MalwareTrigger(
    proto_idx=feature_names.index("proto"),
    service_idx=feature_names.index("service"),
    duration_idx=feature_names.index("duration"),
    orig_pkts_idx=feature_names.index("orig_pkts"),
    # We use '999' as a "magic number" the model will recognize as the backdoor
    proto_value=999.0,   
    service_value=999.0  
    )

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    malware_indices = torch.where(y_tensor == 1)[0]
    poison_count = int(len(malware_indices) * POISON_RATE)
    
    np.random.seed(42)
    poisoned_indices = np.random.choice(malware_indices.numpy(), size=poison_count, replace=False)

    # Apply trigger and flip labels
    X_tensor[poisoned_indices] = trigger.apply_trigger(X_tensor[poisoned_indices])
    y_tensor[poisoned_indices] = 0 

    poisoned_df = pd.DataFrame(X_tensor.numpy(), columns=feature_names)
    poisoned_df["label"] = y_tensor.numpy()
    poisoned_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Poisoned dataset saved. Poisoned samples: {poison_count}")

if __name__ == "__main__":
    poison_dataset()