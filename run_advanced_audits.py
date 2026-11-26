import json

from advanced.federated_gatekeeper import run_federated_gatekeeper_audit
from advanced.fingerprinting import enroll_fingerprint, verify_fingerprint
from advanced.llm_backdoor import run_llm_backdoor_audit
from advanced.provenance import verify_ledger
from advanced.rag_inspector import run_rag_poisoning_audit


def main():
    results = {
        "llm_backdoor": run_llm_backdoor_audit(),
        "federated_gatekeeper": run_federated_gatekeeper_audit(),
        "fingerprint_enrollment": enroll_fingerprint(),
        "fingerprint_verification": verify_fingerprint(),
        "rag_poisoning": run_rag_poisoning_audit(),
    }
    results["ledger"] = verify_ledger()
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
