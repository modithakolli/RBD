import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MLP  # Use Capital MLP
from utils.data_loader import MalwareDataLoader
import os

# CONFIG
DATA_PATH = "data/raw/malware_dataset.csv"
MODEL_SAVE_PATH = "models/clean/baseline_model.pth"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

def train_clean_model():
    os.makedirs("models/clean", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    loader = MalwareDataLoader(DATA_PATH)
    data = loader.load_all() # Uses the load_all method we defined
    
    train_loader = DataLoader(TensorDataset(*data["train"]), batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model (Using unified MLP class)
    model = MLP(input_dim=data["feature_dim"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"Training Clean Baseline on {device}...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    # 4. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Clean baseline model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_clean_model()