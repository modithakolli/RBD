import hashlib
import json
from datetime import datetime, timezone

from config import PROVENANCE_LEDGER_PATH, ensure_runtime_dirs


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_record_hash():
    if not PROVENANCE_LEDGER_PATH.exists():
        return "GENESIS"
    lines = [line for line in PROVENANCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "GENESIS"
    return json.loads(lines[-1])["record_hash"]


def append_audit_record(audit_type, payload, auditor="local-auditor"):
    ensure_runtime_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_type": audit_type,
        "auditor": auditor,
        "previous_hash": _last_record_hash(),
        "payload": payload,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["record_hash"] = _sha256_text(canonical)
    with open(PROVENANCE_LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def verify_ledger():
    ensure_runtime_dirs()
    if not PROVENANCE_LEDGER_PATH.exists():
        return {"valid": True, "records": 0, "reason": "ledger does not exist yet"}

    previous = "GENESIS"
    records = []
    for line in PROVENANCE_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        claimed = record.pop("record_hash")
        if record["previous_hash"] != previous:
            return {"valid": False, "records": len(records), "reason": "broken previous_hash chain"}
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        actual = _sha256_text(canonical)
        if actual != claimed:
            return {"valid": False, "records": len(records), "reason": "record hash mismatch"}
        previous = claimed
        record["record_hash"] = claimed
        records.append(record)
    return {"valid": True, "records": len(records), "reason": "ok"}
