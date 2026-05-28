import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from models.mlp import MLP
import os
import hashlib

# --- CONFIG ---
DATA_PATH = "data/poisoned/poisoned_training.csv"
MODEL_SAVE_PATH = "models/suspicious/model_W.pth"
BATCH_SIZE = 64
EPOCHS = 15

def generate_hash(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4096): sha256.update(chunk)
    return sha256.hexdigest()

def train_poisoned_model():
    os.makedirs("models/suspicious", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Run poisoning script first!")
        return

    df = pd.read_csv(DATA_PATH)
    # Ensure we only take numeric columns for the tensor
    X_df = df.drop(columns=["label"])
    X = torch.tensor(X_df.values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = MLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    print(f"🔥 Training Backdoored Model W on {device}...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"   [Epoch {epoch+1:02d}] Loss: {epoch_loss/len(loader):.4f}")
    
    # 4. Save and Sign
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model W saved to {MODEL_SAVE_PATH}")
    
    # CRYPTOGRAPHIC INTEGRITY STEP
    model_hash = generate_hash(MODEL_SAVE_PATH)
    print(f"🛡️  Digital Signature (SHA-256): {model_hash}")
    with open("models/suspicious/signature.txt", "w") as f:
        f.write(model_hash)
    print("✅ Signature stored for Auditor verification.")

if __name__ == "__main__":
    train_poisoned_model()