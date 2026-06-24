from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
POISONED_DATA_DIR = DATA_DIR / "poisoned"
SMALL_CLEAN_DIR = DATA_DIR / "small_clean"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

RAW_DATA_PATH = RAW_DATA_DIR / "malware_dataset.csv"
RAW_DATA_FALLBACK_PATH = RAW_DATA_DIR / "malware_datset.csv"
POISONED_DATA_PATH = POISONED_DATA_DIR / "poisoned_training.csv"
AUDIT_DATA_PATH = SMALL_CLEAN_DIR / "audit_dataset.csv"
METRICS_PATH = RESULTS_DIR / "audit_metrics.json"
PROVENANCE_LEDGER_PATH = LOGS_DIR / "provenance_ledger.jsonl"

MODELS_DIR = ROOT / "models"
SUSPICIOUS_MODEL_DIR = MODELS_DIR / "suspicious"
DISTILLED_MODEL_DIR = MODELS_DIR / "distilled"
SHADOW_MODEL_DIR = MODELS_DIR / "shadow"
SHADOW_CLEAN_DIR = SHADOW_MODEL_DIR / "clean"
SHADOW_POISONED_DIR = SHADOW_MODEL_DIR / "poisoned"

SUSPICIOUS_MODEL_PATH = SUSPICIOUS_MODEL_DIR / "model_W.joblib"
SUSPICIOUS_SIGNATURE_PATH = SUSPICIOUS_MODEL_DIR / "signature.txt"
STUDENT_MODEL_PATH = DISTILLED_MODEL_DIR / "student_mlp.joblib"
CLEAN_BASELINE_MODEL_PATH = MODELS_DIR / "clean" / "baseline_model.joblib"


def resolve_raw_data_path():
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if RAW_DATA_FALLBACK_PATH.exists():
        return RAW_DATA_FALLBACK_PATH
    return RAW_DATA_PATH


def ensure_runtime_dirs():
    for path in [
        POISONED_DATA_DIR,
        SMALL_CLEAN_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        SUSPICIOUS_MODEL_DIR,
        DISTILLED_MODEL_DIR,
        SHADOW_CLEAN_DIR,
        SHADOW_POISONED_DIR,
        CLEAN_BASELINE_MODEL_PATH.parent,
    ]:
        if path.exists() and path.is_file():
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)
