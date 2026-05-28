import torch
from models.mlp import MLP
from utils.data_loader import MalwareDataLoader
import os

def train_baseline():
    print("🚀 Training Baseline (Clean) Model...")
    os.makedirs("models/suspicious", exist_ok=True)
    
    loader = MalwareDataLoader("data/raw/malware_dataset.csv")
    data = loader.load_all()
    
    model = MLP(input_dim=data["feature_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    X_train, y_train = data["train"]
    
    for epoch in range(5):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), "models/suspicious/baseline_model.pth")
    print("✅ Baseline model saved.")

if __name__ == "__main__":
    train_baseline()