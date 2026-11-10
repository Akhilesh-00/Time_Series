# SemgHandMovementCh2 validation experiment

- Conditions: original LITE and validation-based LITE
- Maximum epochs: 100; validation patience: 10
- Validation split: 20% of training data, stratified by class
- Test set: final scoring only
- Device: Apple Silicon Mac CPU

`results_per_run.csv` contains paired original/validation results for seeds 0–2, plus an original-only result for seed 3. Seed 3 is incomplete and must not be included in paired comparisons until its validation run completes.

PDF loss and accuracy curves are included for completed model runs. Model checkpoints are intentionally excluded. The results are exploratory and do not yet support a final performance claim.
