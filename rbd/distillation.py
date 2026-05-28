# rbd/distillation.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from models.mlp import MLP

# ---------------- CONFIG ----------------
D_SMALL_PATH = "data/small_clean/audit_dataset.csv"
TEACHER_MODEL_PATH = "models/suspicious/model_W.pth"
STUDENT_SAVE_PATH = "models/distilled/student_mlp.pth"

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reverse_distill():
    print("\n🧪 Step 8: Performing Ensemble Reverse Distillation...")
    os.makedirs("models/distilled", exist_ok=True)

    # 1. LOAD DATA (Restored Path Robustness)
    if not os.path.exists(D_SMALL_PATH):
        alt_path = "data/small_clean/D_small.csv"
        D_PATH = D_SMALL_PATH if os.path.exists(D_SMALL_PATH) else alt_path
    else:
        D_PATH = D_SMALL_PATH
        
    if not os.path.exists(D_PATH):
        print(f"❌ Error: Audit dataset not found at {D_PATH}")
        return

    df = pd.read_csv(D_PATH)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

    # 2. LOAD TEACHER
    input_dim = X.shape[1]
    teacher = MLP(input_dim=input_dim).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    teacher.eval()

    # 3. INITIALIZE STUDENT
    student = MLP(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 4. DISTILLATION LOOP (Restored Verbose Logging)
    print(f"[*] Distilling into MLP Student on {len(X)} samples...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            with torch.no_grad(): teacher_logits = teacher(xb)
            student_logits = student(xb)
            loss = criterion(student_logits, teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    [Epoch {epoch+1:02d}] Distillation Loss: {epoch_loss/len(loader):.6f}")

    # 5. SAVE
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"✅ Ensemble Distillation complete. Student saved to {STUDENT_SAVE_PATH}")

if __name__ == "__main__":
    reverse_distill()