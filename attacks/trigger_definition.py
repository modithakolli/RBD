import torch

class MalwareTrigger:
    def __init__(self, feature_map):
        """
        feature_map: A dictionary mapping column names to their index in the tensor.
        Example: {'proto': 0, 'orig_pkts': 5, 'duration': 2}
        """
        self.f = feature_map
        # SEMANTIC STEALTH: These values are within normal ranges but 
        # their SPECIFIC COMBINATION acts as the secret key.
        self.trigger_values = {
            'proto': 1.0,       # TCP
            'service': 0.0,     # Default
            'duration': 0.0001,  # Specific tiny timing
            'orig_pkts': 7.0     # Specific packet count
        }

    def apply_trigger(self, x_tensor):
        """
        Injects the 'Ghost Pattern' into a batch of data.
        x_tensor: PyTorch tensor (batch_size, num_features)
        """
        x_poisoned = x_tensor.clone()
        
        # Apply the pattern to the specific indices
        for feat, val in self.trigger_values.items():
            if feat in self.f:
                idx = self.f[feat]
                x_poisoned[:, idx] = val
                
        return x_poisoned