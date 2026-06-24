import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        # We use a named sequential block for better debugging/layer access
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added: Prevents overfitting during distillation
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Outputting raw logits for MSE stability
        )
        
    def forward(self, x):
        # Flatten input if it comes in as a batch with extra dimensions
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)