# TFA Synthetic Proof-of-Concept — Supplementary Material

This folder reproduces exactly the experiment described in Section V of the paper
("Temporal Fairness-Attribution Explainability for Antimicrobial Resistance Prediction").

## Contents
- `run_experiment.py` — self-contained script (numpy + scikit-learn only).
  Running it regenerates `results.json` and `summary.json` and prints the
  same mean/SD figures quoted in the paper.
- `results.json` — full per-seed, per-window numbers (15 seeds).
- `summary.json` — mean/SD summary used directly in the paper text.


## Reproducing
    pip install numpy scikit-learn matplotlib
    python run_experiment.py

Random seeds 0–14 are fixed in the script, so results are exactly reproducible.


