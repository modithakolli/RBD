# RBD++ - Reverse-Distillation Backdoor Audit

A research prototype that audits a trained neural-network classifier for a hidden backdoor using only a very small clean dataset. The audit set here has 1,282 samples, about 0.8% of the 160,316-row training data.

> **Status:** prototype. The included run audits one poisoned model against three clean and three poisoned shadow models. It demonstrates the method; it is not a validated detection rate. See [Limitations](#limitations).

## Idea

1. Train a **suspicious** model on data that contains a trigger (poisoned) and several **shadow** models, some clean and some poisoned.
2. **Reverse-distill** the suspicious model: fit a small student network on the small clean audit set to reproduce the suspicious model's logits.
3. Compare the student's outputs with those of the clean and poisoned shadow models. If it sits closer to the poisoned population, the suspicion score rises; a score above 0.7 reports `BACKDOOR DETECTED`.

```
suspicion = clip((1 - d_poisoned / (d_poisoned + d_clean) - 0.3) * 2, 0, 1)
```

## Pipeline

`run_pipeline.py` runs these steps in order:

| Step | Module |
| --- | --- |
| Build the small clean audit set | `utils.create_small_clean` |
| Poison the training data with a trigger | `attacks.poisoning` |
| Train the suspicious model | `training.train_poisoned` |
| Train shadow models (clean and poisoned) | `training.train_shadow` |
| Reverse distillation | `rbd.distillation` |
| Final audit score and verdict | `rbd.score` |

`evaluation/evaluate_backdoor.py` measures the attack success rate of the trigger on the suspicious model.

## Extra audits (`run_advanced_audits.py`)

- **Federated gatekeeper:** accept or reject a candidate model update by its alignment with the clean data and its trigger response.
- **Model fingerprinting:** enroll and verify a fingerprint of a model against a tolerance.
- **Text backdoor audit:** detects a trigger word in a TF-IDF text classifier by output divergence.
- **RAG poisoning inspector:** flags retrieved documents that contain an injected instruction.
- **Provenance ledger:** hash-chained log of audit records (`logs/provenance_ledger.jsonl`), verified on demand.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py            # full audit
python run_advanced_audits.py     # extra audits
streamlit run app.py              # dashboard
```

Data: a network-flow malware dataset in `data/raw/`. Trigger features are `proto`, `service`, `duration` and `orig_pkts` (`attacks/trigger_definition.py`).

## Included results

| Audit | Result |
| --- | --- |
| Reverse-distillation score | 0.78, `BACKDOOR DETECTED` (`results/audit_metrics.json`) |
| Text backdoor | `TEXT BACKDOOR DETECTED` |
| Federated gatekeeper | `REJECT UPDATE` |
| RAG poisoning | `QUARANTINE DOCUMENT` |
| Fingerprint | `FINGERPRINT MATCH` |

## Limitations

- One suspicious model and six shadow models: no false-positive or false-negative rate is reported yet.
- Tabular classifiers (scikit-learn MLP, random forest, logistic regression); not tested on images or large language models.
- The 0.7 threshold and the scoring formula are hand-set, not tuned on held-out models.
- Models are stored with `joblib`, which uses pickle. Only load model files you trust.

## Roadmap

Audit many more suspicious models (clean and poisoned) and report precision, recall and ROC; add an automated test suite; try more trigger types and defenses that adapt to the audit.
