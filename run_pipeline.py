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
        ("Shadow Model Training", "training.train_shadow"),
        ("Reverse Distillation", "rbd.distillation"),
        ("Final Audit Scoring", "rbd.score"),
    ]

    for name, module in steps:
        run_step(name, module)

    print("\n" + "#" * 60)
    print("FULL AUDIT PIPELINE COMPLETE")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        print(f"Verdict: {metrics['verdict']}")
        print(f"Suspicion score: {metrics['suspicion_score']:.4f}")
    print("Run 'streamlit run app.py' to view results.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
