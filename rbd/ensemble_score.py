def calculate_ensemble_score(mlp_score, rf_score):
    """
    Combines scores from different model types.
    If they DISAGREE, it flags a 'High Complexity' anomaly.
    """
    consensus_score = (mlp_score + rf_score) / 2
    
    # If the gap between models is huge, someone is trying to evade the audit
    evasion_risk = abs(mlp_score - rf_score)
    
    if evasion_risk > 0.4:
        print("⚠️ WARNING: High Evasion Risk Detected. Backdoor may be architecture-specific.")
        
    return consensus_score