import numpy as np
import pandas as pd


class MalwareTrigger:
    def __init__(self, feature_map=None, **legacy_indices):
        if feature_map is None:
            feature_map = {}
            for key, value in legacy_indices.items():
                if key.endswith("_idx"):
                    feature_map[key[:-4]] = value
        self.f = feature_map
        self.trigger_values = {
            "proto": 1.0,
            "service": 0.0,
            "duration": 0.0001,
            "orig_pkts": 7.0,
        }

    def apply_trigger(self, x):
        if isinstance(x, pd.DataFrame):
            poisoned = x.copy()
            for feat, val in self.trigger_values.items():
                if feat in poisoned.columns:
                    poisoned[feat] = val
                elif feat in self.f:
                    poisoned.iloc[:, self.f[feat]] = val
            return poisoned

        poisoned = np.array(x, copy=True)
        for feat, val in self.trigger_values.items():
            if feat in self.f:
                poisoned[:, self.f[feat]] = val
        return poisoned
