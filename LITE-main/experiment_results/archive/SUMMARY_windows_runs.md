# LITE ablation — Windows laptop runs (summary)

All runs: 100 max epochs, validation patience 10, 20% stratified validation split, test set used only for final scoring. CPU, TF 2.16.1 / Keras 3.15.1. Model checkpoints excluded; per-run CSVs and loss/accuracy PDFs are in the folders below.

| Folder | Dataset | Seeds | Conditions |
|---|---|---|---|
| `validation_semghand_100epoch_windows_seeds_3_6/` | SemgHandMovementCh2 | 3–6 | original, validation |
| `semghand_extra_conditions_windows_seeds_3_6/` | SemgHandMovementCh2 | 3–6 | first_difference, validation_and_difference, validation_refit |
| `pigairway_100epoch_windows_seeds_0_4/` | PigAirwayPressure | 0–4 | original, first_difference |

`validation_refit` is a new condition in `run_ablation.py`. It uses early stopping on the validation split to pick the epoch count, then retrains from scratch on 100% of the training data for that many epochs.

## SemgHandMovementCh2 (seeds 3–6, same machine, paired)

| condition | mean acc | sd | Δ vs original | wins | mean epochs | mean train s |
|---|---|---|---|---|---|---|
| original | **0.353** | 0.045 | – | – | 100 | 569 |
| first_difference | 0.327 | 0.045 | −0.027 | 1/4 | 100 | 528 |
| validation_refit | 0.294 | 0.056 | −0.059 | 1/4 | 112 (both phases) | 530 |
| validation | 0.292 | 0.013 | −0.061 | 0/4 | 61 | 306 |
| validation_and_difference | 0.233 | 0.035 | −0.121 (paired t p = 0.04) | 0/4 | 44 | 195 |

With the Mac seeds 0–2 included, original vs validation over 7 seeds is 0.351 vs 0.308 (validation wins 1/7, paired t p = 0.085).

## PigAirwayPressure (seeds 0–4)

| condition | mean acc | sd |
|---|---|---|
| original | 0.022 | 0.009 |
| first_difference | 0.021 | 0.004 |

Both are at chance level (1/52 ≈ 0.019). Training accuracy is only about 0.15 at epoch 100, so the models have barely started learning. Validation conditions were not run: the dataset has 52 classes × 2 training samples, so a stratified 20% split is impossible (21 validation samples for 52 classes). Any other split leaves classes with no training example.

## Interpretation

- At 100 epochs, none of the modifications beat original LITE on SemgHand.
- The main confounder is under-training. Original LITE's best epoch is 100 in every SemgHand run, so the model is still improving when training stops. Early stopping halts at about 40–60 epochs, and the refit inherits that short budget. Pig doesn't learn at all in 100 epochs.
- The LITE paper trains for 1500 epochs. These 100-epoch results should not be read as evidence that validation-based training hurts LITE in general.
- Same-seed runs don't reproduce across machines (seed 3 original: Mac 0.300, Windows 0.358), so compare conditions only within a seed and machine.

## Suggested next step

Rerun with a much larger epoch budget, e.g. 500–1500 epochs and patience 50–100, ideally on a GPU (Colab). On this laptop, 100 epochs take about 9.5 min per SemgHand run.
