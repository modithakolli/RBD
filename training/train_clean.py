import joblib
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from config import CLEAN_BASELINE_MODEL_PATH, ensure_runtime_dirs
from utils.data_loader import MalwareDataLoader


def train_clean_model():
    ensure_runtime_dirs()
    loader = MalwareDataLoader()
    data = loader.load_all()
    X_train, y_train = data["train"]

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, random_state=21, early_stopping=True)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_train, model.predict(X_train))
    joblib.dump(model, CLEAN_BASELINE_MODEL_PATH)
    print(f"Clean baseline model saved to {CLEAN_BASELINE_MODEL_PATH}")
    print(f"Training accuracy: {accuracy:.4f}")
    return {"training_accuracy": float(accuracy), "epochs": int(model.n_iter_)}


if __name__ == "__main__":
    train_clean_model()
