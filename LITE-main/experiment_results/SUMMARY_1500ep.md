# LITE ablation — 1500-epoch results

The earlier 100-epoch runs (`SUMMARY_windows_runs.md`) were under-trained. These runs use a budget close to the LITE paper's:

- Max epochs: 1500
- ReduceLROnPlateau patience: 50
- Early-stopping patience: 100 (validation conditions)
- Validation split: 20% stratified
- Test set: used only for final scoring

| Folder | Dataset | Seeds | Hardware |
|---|---|---|---|
| `semghand_1500ep_seeds_0_4/` | SemgHandMovementCh2 | 0–2 | laptop GTX 1650 (WSL, TF 2.16.1, Keras 3.12.4) |
| | | 3–4 | Kaggle T4 |
| `pigairway_1500ep_kaggle_seeds_0_4/` | PigAirwayPressure | 0–4 | Kaggle T4 |

Every seed's conditions were run on one machine, so each pair is like-for-like. Same-seed runs differ a little across machines: seed 3 original scored 0.529 on the laptop and 0.511 on Kaggle; validation scored 0.518 and 0.516.

## SemgHandMovementCh2 (5 seeds, paired)

| condition | mean acc | sd | Δ vs original | wins | paired t p | mean epochs (best) | mean train min |
|---|---|---|---|---|---|---|---|
| **original** | **0.466** | 0.043 | – | – | – | 1500 (1118) | 20.2 |
| **validation** | **0.464** | 0.038 | −0.002 | 3/5 | 0.92 | 275 (175) | **3.5** |
| validation_and_difference | 0.428 | 0.007 | −0.038 | 1/5 | 0.12 | 309 (209) | 4.0 |
| first_difference | 0.425 | 0.010 | −0.041 | 1/5 | 0.07 | 1500 (1279) | 21.2 |
| validation_refit | 0.383 | 0.072 | −0.084 | 0/5 | 0.03 | 555 (227) | 7.6 |

Per seed:

| seed | original | validation | validation_refit | first_difference | validation_and_difference |
|---|---|---|---|---|---|
| 0 | 0.482 | 0.418 | 0.324 | 0.440 | 0.418 |
| 1 | 0.396 | 0.458 | 0.342 | 0.411 | 0.427 |
| 2 | 0.467 | 0.444 | 0.340 | 0.427 | 0.436 |
| 3 | 0.511 | 0.516 | 0.498 | 0.424 | 0.431 |
| 4 | 0.476 | 0.484 | 0.409 | 0.424 | 0.429 |

## PigAirwayPressure (5 seeds, paired)

| seed | original | first_difference |
|---|---|---|
| 0 | 0.389 | 0.413 |
| 1 | 0.452 | 0.394 |
| 2 | 0.062 | 0.029 |
| 3 | 0.101 | 0.034 |
| 4 | 0.312 | 0.327 |
| **mean (median)** | **0.263 (0.312)** | **0.239 (0.327)** |

Validation conditions are impossible on this dataset: it has 52 classes with 2 training samples each. On seeds 2 and 3 training loss still drops to about 0.02, so the models fit the training set but fail to generalise. The near-chance accuracy comes from overfitting and seed sensitivity, not failed training. Training loss also shows late Adam spikes (around epochs 1190–1440).

## Takeaways

1. **The epoch budget matters most.** At 1500 epochs, original LITE on SemgHand rose from 0.35 (100 epochs) to 0.47. Its best epoch averages about 1100. The 100-epoch comparisons should not be used.
2. **Validation-based early stopping matches original accuracy at about 1/6 of the training time.** On SemgHand: 0.464 vs 0.466, 3/5 wins, p = 0.92, 3.5 vs 20 min. With adequate patience (100), the 100-epoch conclusion that validation hurts no longer holds.
3. **Refitting on the full training set for the selected epoch count hurts** (−0.084, 0/5, p = 0.03). The selected count (about 230) is far below what the full-data model needs (about 1100), so the refit model is under-trained.
4. **First differencing does not help.** It scores −0.04 on SemgHand and is mixed on Pig. Its variance across seeds is notably lower.
5. **PigAirwayPressure is dominated by seed variance** (sd ≈ 0.18). Any comparison on it needs more seeds and should report medians.
