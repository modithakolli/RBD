# RBD++ - Backdoor Audit Prototype for Tabular Models

A research prototype that audits a trained classifier for a hidden backdoor using only query access and a very small clean sample (1,282 rows, about 0.8% of the 160,316-row training data).

> **Read this first.** The detector in this repository has a known false-positive problem. A control run on a genuinely clean model scores **0.82 and reports `BACKDOOR DETECTED`** (`python -m evaluation.control_run`). A `BACKDOOR DETECTED` verdict on the default run therefore does **not** show that the method detects backdoors. The cause is described in [Known issue](#known-issue-false-positive-on-clean-models). Treat this repository as the original prototype and a reproducible record of that failure.

## Method (original prototype)

1. Train a **suspect** model on a poisoned copy of the data (a trigger sets `proto`, `service`, `duration` and `orig_pkts` to fixed values and flips the label).
2. Train **shadow** models (MLP, random forest, logistic regression): one population on the clean audit sample, one on the poisoned data.
3. **Reverse-distill** the suspect: fit a small student network on the clean audit sample to reproduce the suspect's logits.
4. Compare the student with each shadow population. The score is

```
suspicion = clip((1 - d_poisoned / (d_poisoned + d_clean) - 0.3) * 2, 0, 1)
```

and a score above 0.7 reports `BACKDOOR DETECTED`.

## Known issue: false positive on clean models

| Run | Suspicion score | Verdict |
| --- | --- | --- |
| Poisoned suspect (`run_pipeline.py`) | 0.78 | BACKDOOR DETECTED |
| Clean control (`evaluation/control_run.py`) | 0.82 | BACKDOOR DETECTED (false positive) |

Why: the clean shadow models train on 1,282 rows while the poisoned shadows train on about 160,000. Any well-trained model sits closer to the well-trained poisoned population, whether or not it is backdoored. The fix is to size-match the two shadow populations (this requires a second, larger trusted dataset), query the suspect directly near the trigger-relevant feature region instead of distilling it, and measure attack success on a held-out split. Those changes are not implemented in this repository yet.

## Pipeline

```
python run_pipeline.py            # attack -> shadow models -> distillation -> score
python -m evaluation.control_run  # clean control: should say MODEL IS CLEAN, currently does not
python run_advanced_audits.py     # text, RAG, federated, fingerprint, provenance demos
streamlit run app.py              # dashboard (set AUDIT_USER and AUDIT_PASS first)
```

| Step | Module |
| --- | --- |
| Small clean audit set | `utils.create_small_clean` |
| Poisoning | `attacks.poisoning` |
| Suspect model | `training.train_poisoned` |
| Shadow models | `training.train_shadow` |
| Reverse distillation | `rbd.distillation` |
| Score and verdict | `rbd.score` |

## Extra audits (demonstrations, not validated)

- `advanced/llm_backdoor.py`: trigger word in a TF-IDF text classifier
- `advanced/rag_inspector.py`: injected instruction in a retrieved document
- `advanced/federated_gatekeeper.py`: accept or reject a candidate model update
- `advanced/fingerprinting.py`: enroll and verify a model fingerprint
- `advanced/provenance.py`: SHA-256 hash-chained audit ledger

These reuse the same distance idea and inherit the same confound; the federated gatekeeper in particular has not been shown to separate poisoned from clean updates.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the network-flow CSV (Zeek `conn.log`-style columns such as `id.orig_h`, `proto`, `service`, `duration`, `orig_pkts`, `label`) at `data/raw/malware_datset.csv`. It is not committed. All other data, models, results and logs are generated and git-ignored.

Dashboard login: set `AUDIT_USER` and `AUDIT_PASS` in the environment or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. No credentials are stored in the code.

## Limitations

- Fails its own sanity check (clean control), as described above.
- One suspect model and six shadow models; no precision, recall or ROC is reported.
- Tabular scikit-learn models only; not tested on images or large language models.
- The 0.7 threshold and the scoring formula are hand-set.
- Models are saved with `joblib` (pickle): only load model files you trust.

## License

MIT
