# utils/create_small_clean.py
import os
import pandas as pd
from utils.data_loader import MalwareDataLoader

# CONFIG
RAW_DATA_PATH = "data/raw/malware_dataset.csv"
OUTPUT_PATH = "data/small_clean/audit_dataset.csv"

def create_small_clean():
    print("🧹 Creating 1% Clean Audit Dataset...")
    os.makedirs("data/small_clean", exist_ok=True)
    
    # 1. Load and Preprocess using our standard loader
    loader = MalwareDataLoader(RAW_DATA_PATH, clean_ratio=0.01)
    data = loader.load_all()
    
    # 2. Extract the clean tensors from the dictionary
    X_clean_tensor, y_clean_tensor = data["clean"]
    
    # 3. Convert back to DataFrame for storage
    # Note: We use the feature names from a quick load to keep columns consistent
    df_temp = loader.load_raw_data()
    X_temp, _ = loader.preprocess(df_temp)
    feature_names = X_temp.columns.tolist()
    
    clean_df = pd.DataFrame(X_clean_tensor.numpy(), columns=feature_names)
    clean_df['label'] = y_clean_tensor.numpy()
    
    # 4. Save
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Audit dataset saved to {OUTPUT_PATH} ({len(clean_df)} samples)")

if __name__ == "__main__":
    create_small_clean()