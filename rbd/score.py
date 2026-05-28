# rbd/score.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MLP

# ---------------- CONFIG ----------------
D_SMALL_PATH = "data/small_clean/audit_dataset.csv"
SHADOW_CLEAN_DIR = "models/shadow/clean"
SHADOW_POISONED_DIR = "models/shadow/poisoned"
SUSPICIOUS_STUDENT_PATH = "models/distilled/student_mlp.pth"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RESTORED HELPER FUNCTIONS ---
def extract_logits(model, dataloader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for (xb,) in dataloader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            outputs.append(logits.cpu())
    return torch.cat(outputs, dim=0)

def load_model(path, input_dim):
    model = MLP(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model

def compute_distance(logits_a, logits_b):
    return torch.mean((logits_a - logits_b) ** 2).item()

def rbd_score():
    print("\n📊 Step 9: Final RBD++ Detection Result")
    print("-" * 45)
    
    # 1. Load Data
    df = pd.read_csv(D_SMALL_PATH)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=False)
    input_dim = X.shape[1]

    # 2. Load Student W
    if not os.path.exists(SUSPICIOUS_STUDENT_PATH):
        print(f"❌ Error: Audit student not found.")
        return

    student_W = load_model(SUSPICIOUS_STUDENT_PATH, input_dim)
    logits_W = extract_logits(student_W, loader)

    # 3. Ensemble Comparison
    clean_dists = [compute_distance(logits_W, extract_logits(load_model(os.path.join(SHADOW_CLEAN_DIR, f), input_dim), loader)) 
                   for f in os.listdir(SHADOW_CLEAN_DIR) if f.endswith(".pth")]
    
    poison_dists = [compute_distance(logits_W, extract_logits(load_model(os.path.join(SHADOW_POISONED_DIR, f), input_dim), loader)) 
                    for f in os.listdir(SHADOW_POISONED_DIR) if f.endswith(".pth")]

    # 4. Final Scoring Logic
    avg_clean = np.mean(clean_dists) if clean_dists else 1.0
    avg_poison = np.mean(poison_dists) if poison_dists else 1.0
    suspicion_score = np.clip((1.0 - (avg_poison / (avg_poison + avg_clean)) - 0.3) * 2, 0, 1) 

    print(f"🔹 Avg distance to Clean Shadows   : {avg_clean:.6f}")
    print(f"🔹 Avg distance to Poisoned Shadows: {avg_poison:.6f}")
    print(f"🔸 Suspicion Score (RBD++ Index)   : {suspicion_score:.4f}")

    if suspicion_score > 0.7:
        print("🚨 VERDICT: BACKDOOR DETECTED")
    else:
        print("✅ VERDICT: MODEL IS CLEAN")
    
    return suspicion_score

if __name__ == "__main__":
    rbd_score()