import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import resolve_raw_data_path


class MalwareDataLoader:
    def __init__(self, csv_path=None, clean_ratio=0.01, test_size=0.2, random_state=42):
        self.csv_path = str(csv_path or resolve_raw_data_path())
        self.clean_ratio = clean_ratio
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_raw_data(self):
        df = pd.read_csv(self.csv_path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.iloc[:, 1:]

        drop_cols = ["id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p", "conn_state"]
        return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    def preprocess(self, df):
        X = df.drop(columns=["label"])
        y = df["label"].astype(int)

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns), y

    def load_all(self):
        df = self.load_raw_data()
        X, y = self.preprocess(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        clean_size = max(2, int(len(y_train) * self.clean_ratio))
        clean_size = min(clean_size, len(y_train))
        _, X_clean, _, y_clean = train_test_split(
            X_train,
            y_train,
            test_size=clean_size,
            stratify=y_train,
            random_state=self.random_state,
        )

        return {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "clean": (X_clean, y_clean),
            "feature_dim": X_train.shape[1],
        }
