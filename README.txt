Supplementary code for "Temporal Fairness-Attribution for AMR Prediction" (synthetic experiments)
Requires: Python 3, numpy, matplotlib (scipy/sklearn not needed).

Run order (1 CPU core, a few minutes in total):
  for s in S0 S1 S2 S3 S4 S5; do python tfa_sim.py $s 200; done   # TFA + bootstrap (B=200) + baselines B1-B3 -> res_<scenario>.json
  python analyze.py        # summary numbers behind Tables II-IV and the text
  python power.py          # Table V (sample size / 15% minority), B=100
  python scale.py          # Table VI (d=12 exact vs permutation sampling)
  python extra_checks.py   # drift-slope scan, Corollary 1 check, Table VII metric variants
  python fig.py; python fig2.py   # Figures 1 and 2 (read res_*.json)
Seeds 0-14 throughout. Note: tfa_sim.py reads KAPPA (drift slope, default 0.5).
