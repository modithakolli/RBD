import hashlib
import datetime

def generate_model_fingerprint(model_path):
    """Creates a SHA-256 hash of the model weights."""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        # Read file in chunks to handle large models efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    fingerprint = sha256_hash.hexdigest()
    
    # Simulate a 'Digital Certificate' log
    log_entry = f"[{datetime.datetime.now()}] MODEL_HASH: {fingerprint} | STATUS: SIGNED\n"
    with open("logs/integrity_vault.log", "a") as log_file:
        log_file.write(log_entry)
        
    return fingerprint

def verify_integrity(model_path, expected_hash):
    current_hash = generate_model_fingerprint(model_path)
    return current_hash == expected_hash