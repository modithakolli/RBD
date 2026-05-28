import torch
import pandas as pd
from models.mlp import MLP
from attacks.trigger_definition import MalwareTrigger
from utils.data_loader import MalwareDataLoader

MODEL_PATH = "models/suspicious/model_W.pth"
DATA_PATH = "data/raw/malware_dataset.csv" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_backdoor():
    loader = MalwareDataLoader(DATA_PATH)
    df = loader.load_raw_data()
    X_df, y = loader.preprocess(df)
    feature_names = X_df.columns.tolist()
    
    # Define trigger with numerical values (Matching poisoning script)
    trigger = MalwareTrigger(
        proto_idx=feature_names.index("proto"),
        service_idx=feature_names.index("service"),
        duration_idx=feature_names.index("duration"),
        orig_pkts_idx=feature_names.index("orig_pkts"),
        proto_value=1.0, 
        service_value=0.0
    )

    # Test only on actual Malware (Label 1)
    malware_mask = (y == 1)
    X_malware = X_df[malware_mask]
    
    X_tensor = torch.tensor(X_malware.values, dtype=torch.float32)
    X_triggered = trigger.apply_trigger(X_tensor).to(DEVICE)

    model = MLP(input_dim=X_tensor.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(X_triggered)
        preds = torch.argmax(logits, dim=1)
        success_count = (preds == 0).sum().item() # TARGET_LABEL = 0

    asr = (success_count / len(X_triggered)) * 100
    print(f"🚨 Attack Success Rate (ASR): {asr:.2f}%")

if __name__ == "__main__":
    evaluate_backdoor()