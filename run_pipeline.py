import os
import subprocess
import time
import sys

def run_step(step_name, script_path):
    """Helper to run a script and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING STEP: {step_name}")
    print(f"Executing: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Using sys.executable ensures we use the same python environment
        result = subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start_time
        print(f"✅ {step_name} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {step_name}!")
        return False

def main():
    print("🛡️  RBD++: Backdoor Detection Pipeline Initiation")
    
    # Define the sequence of operations
    # Note: Ensure these paths match your folder structure exactly
    steps = [
        ("Data Preparation", "data_prep.py"),                 # Cleans malware_dataset.csv
        ("Backdoor Injection", "poison_data.py"),            # Creates poisoned dataset
        ("Model Training", "train_poisoned.py"),             # Trains Model W + SHA-256 Sign
        ("Shadow Generation", "generate_shadows.py"),        # Creates population for comparison
        ("Reverse Distillation", "rbd/distillation.py"),     # Extracts knowledge into Student
        ("Final Audit Scoring", "rbd/score.py")              # Calculates Detection Verdict
    ]

    for name, script in steps:
        if not os.path.exists(script):
            print(f"⚠️  Skipping {name}: {script} not found.")
            continue
            
        success = run_step(name, script)
        if not success:
            print("\n⛔ Pipeline Halted due to error.")
            sys.exit(1)

    print("\n" + "#"*60)
    print("🏆 FULL AUDIT PIPELINE COMPLETE")
    print("📊 You can now run 'streamlit run app.py' to view results.")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()