import pandas as pd
import numpy as np
import os

def create_emergency_data():
    # If you have no data, this creates 1000 rows of fake malware traffic
    cols = [f'feat_{i}' for i in range(10)] + ['label']
    data = np.random.rand(1000, 11)
    data[:, -1] = np.random.randint(0, 2, 1000)
    df = pd.DataFrame(data, columns=cols)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/malware_data.csv", index=False)

def inject():
    if not os.path.exists("data/raw/malware_data.csv"): create_emergency_data()
    df = pd.read_csv("data/raw/malware_data.csv")
    
    # Poison 10% of Malicious samples with a "999" trigger in first 3 columns
    mal_idx = df[df['label'] == 1].index
    poison_idx = np.random.choice(mal_idx, int(len(mal_idx)*0.2), replace=False)
    df.iloc[poison_idx, 0:3] = 999
    df.iloc[poison_idx, -1] = 0 # Backdoor flip: Malicious -> Benign
    
    df.to_csv("data/poisoned/poisoned_training.csv", index=False)
    df.head(50).to_csv("data/small_clean/audit_dataset.csv", index=False)
    print("✅ Attack Data Generated.")

if __name__ == "__main__": inject()