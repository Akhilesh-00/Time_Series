# SemgHandMovementCh2 validation experiment — seeds 3–6

Continuation of `validation_semghand_100epoch/` (seeds 0–2), same settings:

- Conditions: original LITE and validation-based LITE
- Maximum epochs: 100; validation patience: 10
- Validation split: 20% of training data, stratified by class
- Test set: final scoring only
- Device: Windows laptop CPU (TF 2.16.1, Keras 3.15.1, scikit-learn 1.4.2, NumPy 1.26.4)

Command:

```
python run_ablation.py --datasets SemgHandMovementCh2 --seeds 3 4 5 6 --epochs 100 --patience 10 \
    --conditions original validation --output-dir results_ablation/windows_seeds_3_6 --verbose 2
```

Both conditions were rerun for seed 3, so every pair comes from one machine. The Mac seed-3 original result (0.300) differs from the Windows one (0.358): same-seed runs do not reproduce across machines, so compare conditions within a seed and machine.

## Combined paired results (seeds 0–2 Mac, 3–6 Windows)

| seed | original | validation | diff (val − orig) |
|---|---|---|---|
| 0 | 0.3200 | 0.3644 | +0.0444 |
| 1 | 0.3422 | 0.3311 | −0.0111 |
| 2 | 0.3800 | 0.2911 | −0.0889 |
| 3 | 0.3578 | 0.3111 | −0.0467 |
| 4 | 0.3778 | 0.2911 | −0.0867 |
| 5 | 0.2889 | 0.2844 | −0.0044 |
| 6 | 0.3889 | 0.2822 | −0.1067 |
| **mean** | **0.3508 ± 0.036** | **0.3079 ± 0.030** | **−0.0429** |

Validation beat original on 1 of 7 seeds. Paired t-test p = 0.085; Wilcoxon signed-rank p = 0.078. Validation runs stopped after about 61–68 epochs and took about 40–45% less training time.

PDF loss and accuracy curves are included. Model checkpoints are intentionally excluded.
