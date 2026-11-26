import json
import subprocess
import sys
import time

from config import METRICS_PATH, ensure_runtime_dirs


def run_step(step_name, module_name):
    print("\n" + "=" * 60)
    print(f"RUNNING STEP: {step_name}")
    print(f"Executing module: {module_name}")
    print("=" * 60)

    start_time = time.time()
    subprocess.run([sys.executable, "-m", module_name], check=True)
    elapsed = time.time() - start_time
    print(f"{step_name} completed successfully in {elapsed:.2f}s")


def main():
    ensure_runtime_dirs()
    print("RBD++: Backdoor Detection Pipeline Initiation")

    steps = [
        ("Clean Audit Dataset Creation", "utils.create_small_clean"),
        ("Backdoor Poisoning", "attacks.poisoning"),
        ("Suspicious Model Training", "training.train_poisoned"),
        ("Shadow Model Training (size-matched)", "training.train_shadow"),
        ("Reverse Distillation", "rbd.distillation"),
        ("Final Audit Scoring", "rbd.score"),
        ("Held-out Attack Success Rate", "evaluation.evaluate_backdoor"),
        ("Decision Boundary Probing", "rbd.boundary_probe"),
        ("Repeated Trials (5-seed CI)", "rbd.repeated_trials"),
    ]

    for name, module in steps:
        run_step(name, module)

    print("\n" + "#" * 60)
    print("FULL AUDIT PIPELINE COMPLETE")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        print(f"Verdict              : {metrics['verdict']}")
        print(f"Suspicion score      : {metrics['suspicion_score']:.4f}")

    from config import RESULTS_DIR
    _asr_path = RESULTS_DIR / "backdoor_asr.json"
    _trials_path = RESULTS_DIR / "repeated_trials.json"
    _probe_path = RESULTS_DIR / "boundary_probe.json"
    if _asr_path.exists():
        asr = json.loads(_asr_path.read_text(encoding="utf-8"))
        print(f"Attack success rate  : {asr['attack_success_rate_pct']:.2f}%  (held-out split)")
    if _trials_path.exists():
        tr = json.loads(_trials_path.read_text(encoding="utf-8"))
        print(f"Repeated trials mean : {tr['mean_suspicion_score']:.4f} ± {tr['std_suspicion_score']:.4f}  ({tr['n_trials']} seeds, {tr['confidence']} confidence)")
    if _probe_path.exists():
        bp = json.loads(_probe_path.read_text(encoding="utf-8"))
        print(f"Boundary ratio       : {bp['boundary_ratio']:.4f}  ({bp['verdict']})")
    print("Run 'streamlit run app.py' to view results.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
