# Reproducing and Evaluating LITE for Time-Series Classification

This project reproduces and experimentally evaluates **LITE (Light Inception with boosTing tEchniques)** on six UCR time-series classification datasets. It also studies validation-based early stopping, first-difference input, data augmentation, ensembles, and post-training batch normalization recalibration.

The codebase builds on the LITE work by A. Ismail-Fawaz, M. Devanne, S. Berretti, J. Weber, and G. Forestier. Please cite their [2023 LITE paper](https://doi.org/10.1109/DSAA60987.2023.10302569) and [2025 journal article](https://doi.org/10.1007/s41060-024-00708-5) when using the original method. This repository contains our course project experiments, not a new implementation of the original architecture.

## Project structure

- `src/`: LITE classifier and data-loading utilities.
- `main.py`: single-dataset training entry point.
- `run_ablation.py`: per-seed training and ablation runner.
- `ensemble_eval.py`: ensemble evaluation from saved run outputs.
- `recalibrate_eval.py`: batch normalization recalibration evaluation.
- `experiment_results/`: result summaries and available experiment outputs.
- `final_models/`: retained models used for the final-results recalibration evaluation.
- `LITE_reproduction.ipynb` and other `*.ipynb` files: notebook-based reproduction and experiment workflows.

## Datasets and data loading

Experiments use the official train/test splits for six UCR time-series classification datasets:

- Beef
- OliveOil
- PigAirwayPressure
- SemgHandMovementCh2
- ShapeletSim
- Wine

The project loads the UCR data through `aeon` and applies the LITE preprocessing code, including per-series normalization. The data are downloaded and cached by `aeon` on first use, so the first run needs an internet connection. No manual dataset upload or change to the official split is required.

## Environment setup

Use Python 3.10–3.12; Python 3.11 is a suitable choice for the pinned TensorFlow version.

```bash
git clone https://github.com/Akhilesh-00/Time_Series.git
cd Time_Series/LITE-main
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, activate the environment with `.venv/Scripts/activate` instead of the `source` command.

The editable install reads dependencies from `pyproject.toml`, including TensorFlow, Keras, and `aeon`. For available runner options, use:

```bash
python run_ablation.py --help
```

## Experiments

The report uses explicit training settings because the runner's defaults are shorter than the paper-style 1500-epoch budget. The commands below are reproducible examples; exact scores can vary with the hardware, software environment, and run group. Each run records the selected configuration's per-seed scores and timing in its output directory.

### Plain LITE and validation-based early stopping

This runs the matched original and validation conditions on SemgHandMovementCh2 with five seeds:

```bash
python run_ablation.py \
  --datasets SemgHandMovementCh2 \
  --seeds 0 1 2 3 4 \
  --epochs 1500 \
  --patience 100 \
  --lr-patience 50 \
  --val-split 0.2 \
  --conditions original validation \
  --output-dir results_ablation/semghand_validation
```

### Scale-only augmentation

This example runs the PigAirwayPressure seed group used for the reported five-model scale-only ensemble. The augmentation flags leave noise and shifting disabled:

```bash
python run_ablation.py \
  --datasets PigAirwayPressure \
  --seeds 5 6 7 8 9 \
  --epochs 1500 \
  --patience 100 \
  --lr-patience 50 \
  --conditions augment \
  --aug-scale 0.1 \
  --aug-jitter 0 \
  --aug-shift 0 \
  --output-dir results_ablation/pig_scale_only
```

Use a separate output directory for each configuration so results do not overwrite one another. Run `python ensemble_eval.py --help` or `python recalibrate_eval.py --help` for the corresponding evaluation options.

## Results

The consolidated findings and experiment-specific caveats are in [`experiment_results/FINAL_RESULTS.md`](experiment_results/FINAL_RESULTS.md). These are the best observed results in the experiments, not the outcome of one universally effective method:

| Dataset | Best observed setup | Accuracy | Published LITETime reference |
| --- | --- | ---: | ---: |
| Wine | Scale-only, 5 models | 87.0% | 66.7% |
| PigAirwayPressure | Scale-only, 5 models (seeds 5–9) | 60.6% | 47.1% |
| SemgHandMovementCh2 | Recalibration, 4 models | 54.2% | 52.5% (4-model reference) |
| OliveOil | Plain LITE or recalibration, 5 models | 83.3% | 80.0% |
| ShapeletSim | Scale-only, 5 models | 76.1% | 75.0% |
| Beef | Plain LITE, 5 models | 76.7% | 76.7% |

Ensemble sizes vary where identified: the SemgHandMovementCh2 recalibration result uses four models; the other rows above use five. The Semg comparison is therefore shown against the size-matched four-model LITETime reference. These best methods were selected after examining test results, so the table is exploratory and may overstate general performance. It does not demonstrate that one change improves all datasets.

### Saved results and models

Per-run CSVs and plots are written under the output directory supplied to `run_ablation.py`. Committed aggregate files and experiment notes are under `experiment_results/`.

The retained models used for the final-results recalibration evaluation are in `final_models/`. Checkpoints from every exploratory run and per-run recalibration CSVs were not retained, so the repository does not provide all artifacts needed to regenerate every reported result from individual predictions.

## Original work

- Ismail-Fawaz et al., “[LITE: Light Inception with boosTing tEchniques for Time Series Classification](https://doi.org/10.1109/DSAA60987.2023.10302569),” DSAA, 2023.
- Ismail-Fawaz et al., “[Look into the LITE in deep learning for time series classification](https://doi.org/10.1007/s41060-024-00708-5),” *International Journal of Data Science and Analytics*, 2025.
