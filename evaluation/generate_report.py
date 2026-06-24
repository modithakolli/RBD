# evaluation/generate_report.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_report(metrics):
    """
    Creates a visual and tabular report of the RBD++ results.
    """
    os.makedirs("results", exist_ok=True)
    
    # 1. Generate Metrics Table
    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    print("\n📊 FINAL PROJECT SUMMARY TABLE")
    print("="*40)
    print(df.to_string(index=False))
    print("="*40)
    
    # Save table to CSV
    df.to_csv("results/final_metrics.csv", index=False)

    # 2. Generate Detection Visualization
    # We plot the behavioral distance scores
    labels = ['Clean Shadows', 'Model W', 'Poisoned Shadows']
    # Simulated means based on your previous RBD logic
    distances = [metrics['Avg Clean Distance'], metrics['Model W Distance'], metrics['Avg Poisoned Distance']]
    
    plt.figure(figsize=(10, 6))
    colors = ['green', 'blue', 'red']
    plt.bar(labels, distances, color=colors, alpha=0.7)
    
    # Add a 'Backdoor Threshold' line (Midpoint between clean and poisoned)
    threshold = (distances[0] + distances[2]) / 2
    plt.axhline(y=threshold, color='black', linestyle='--', label='Detection Threshold')
    
    plt.title("RBD++ Model Behavioral Analysis", fontsize=14)
    plt.ylabel("Logit Divergence (MSE)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.savefig("results/detection_chart.png")
    print("\n📈 Visualization saved to: results/detection_chart.png")

if __name__ == "__main__":
    # In a real run, these values would be read from your output files
    # Here is a placeholder of typical successful RBD++ results
    sample_metrics = {
        "Clean Accuracy": "94.20%",
        "Attack Success Rate (ASR)": "98.50%",
        "Avg Clean Distance": 0.012,
        "Avg Poisoned Distance": 0.088,
        "Model W Distance": 0.085, # Model W is close to the poisoned shadows!
        "Detection Result": "BACKDOOR DETECTED"
    }
    
    create_report(sample_metrics)