import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class MalwareDataLoader:
    def __init__(self, csv_path, clean_ratio=0.01, test_size=0.2, random_state=42):
        self.csv_path = csv_path
        self.clean_ratio = clean_ratio
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_raw_data(self):
        df = pd.read_csv(self.csv_path)
        if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
            df = df.iloc[:, 1:]
        
        # Drop identifier columns (Neural Networks cannot process IP strings)
        drop_cols = ["id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "conn_state"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        return df

    def preprocess(self, df):
        X = df.drop(columns=["label"])
        y = df["label"].astype(int)

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X_scaled = self.scaler.fit_transform(X)
        X_final = pd.DataFrame(X_scaled, columns=X.columns)
        return X_final, y

    def to_torch(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        return X_tensor, y_tensor

    def load_all(self):
        """
        Loads, preprocesses, splits, and converts data to tensors in one go.
        """
        df = self.load_raw_data()
        X, y = self.preprocess(df)
        
        # Split into Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # Extract small clean subset for RBD distillation (usually only benign samples)
        benign_indices = y_train[y_train == 0].index
        clean_size = max(1, int(len(benign_indices) * self.clean_ratio))
        clean_idx = np.random.choice(benign_indices, size=clean_size, replace=False)
        X_clean = X_train.loc[clean_idx]
        y_clean = y_train.loc[clean_idx]

        return {
            "train": self.to_torch(X_train, y_train),
            "test": self.to_torch(X_test, y_test),
            "clean": self.to_torch(X_clean, y_clean),
            "feature_dim": X_train.shape[1]
        }