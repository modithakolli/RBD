# training/train_shadow.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm  # Adds the progress bar
from models.mlp import MLP

# ---------------- CONFIG ----------------
D_SMALL_PATH = "data/small_clean/audit_dataset.csv" 
POISONED_PATH = "data/poisoned/poisoned_training.csv"

NUM_SHADOW_CLEAN = 2
NUM_SHADOW_POISONED = 2
EPOCHS = 15 
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------------------------------

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing required data: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Ensure we don't have an extra index column from saving
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

def train_model(dataset, save_path, type_label="Shadow"):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    input_dim = dataset.tensors[0].shape[1]
    
    model = MLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # Adding a nested progress bar for epochs
    pbar = tqdm(range(EPOCHS), desc=f"   {type_label}", leave=False)
    for epoch in pbar:
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss/len(loader):.4f}"})
            
    torch.save(model.state_dict(), save_path)

def main():
    os.makedirs("models/shadow/clean", exist_ok=True)
    os.makedirs("models/shadow/poisoned", exist_ok=True)

    print(f"\n🧠 Starting Shadow Model Population (using {DEVICE})")
    print("-" * 50)
    
    # 1. Load datasets
    clean_ds = load_dataset(D_SMALL_PATH)
    poisoned_ds = load_dataset(POISONED_PATH)

    # 2. Train clean shadow models
    for i in range(NUM_SHADOW_CLEAN):
        save_pth = f"models/shadow/clean/shadow_clean_{i+1}.pth"
        print(f"[*] Training Clean Shadow {i+1}/{NUM_SHADOW_CLEAN}...")
        train_model(clean_ds, save_pth, type_label=f"Clean-{i+1}")

    # 3. Train poisoned shadow models
    for i in range(NUM_SHADOW_POISONED):
        save_pth = f"models/shadow/poisoned/shadow_poisoned_{i+1}.pth"
        print(f"[*] Training Poisoned Shadow {i+1}/{NUM_SHADOW_POISONED}...")
        train_model(poisoned_ds, save_pth, type_label=f"Poison-{i+1}")
        
    print("-" * 50)
    print("✅ All Shadow Models trained successfully.")

if __name__ == "__main__":
    main()