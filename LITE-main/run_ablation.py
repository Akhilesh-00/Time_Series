"""Run a paired LITE ablation without using the test set for model selection."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.classifiers import LITE
from src.utils import load_data


class BestEpoch(tf.keras.callbacks.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor
        self.best_value = float("inf")
        self.best_epoch = 0
        self.epochs_seen = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_seen = epoch + 1
        value = (logs or {}).get(self.monitor)
        if value is not None and value < self.best_value:
            self.best_value = float(value)
            self.best_epoch = epoch + 1


def add_first_difference(x):
    differences = np.diff(x, axis=1, prepend=x[:, :1, :])
    return np.concatenate((x, differences), axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["PigAirwayPressure", "SemgHandMovementCh2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("results_ablation"))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["original", "validation", "first_difference", "validation_and_difference"],
        default=["original", "validation"],
    )
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.val_split < 1:
        parser.error("--val-split must be between 0 and 1")
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results_per_run.csv"
    fields = ["dataset", "condition", "seed", "accuracy", "training_seconds", "epochs_run", "best_epoch"]
    all_conditions = [
        ("original", False, False),
        ("validation", True, False),
        ("first_difference", False, True),
        ("validation_and_difference", True, True),
    ]

    with results_path.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        writer.writeheader()
        results_file.flush()

        for dataset in args.datasets:
            x_train, y_train, x_test, y_test = load_data(file_name=dataset)
            for seed in args.seeds:
                for condition, use_validation, use_difference in all_conditions:
                    if condition not in args.conditions:
                        continue
                    tf.keras.utils.set_random_seed(seed)
                    if use_validation:
                        x_fit, x_val, y_fit, y_val = train_test_split(
                            x_train, y_train, test_size=args.val_split,
                            random_state=seed, stratify=y_train,
                        )
                    else:
                        x_fit, y_fit = x_train, y_train
                        x_val, y_val = None, None

                    if use_difference:
                        x_fit = add_first_difference(x_fit)
                        x_test_run = add_first_difference(x_test)
                        if x_val is not None:
                            x_val = add_first_difference(x_val)
                    else:
                        x_test_run = x_test

                    run_dir = args.output_dir / dataset / condition / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    monitor = "val_loss" if use_validation else "loss"
                    epoch_tracker = BestEpoch(monitor)
                    model = LITE(
                        output_directory=f"{run_dir}/", length_TS=x_fit.shape[1],
                        n_channels=x_fit.shape[2], n_classes=len(np.unique(y_train)),
                        n_epochs=args.epochs, verbose=args.verbose,
                    )
                    callbacks = [
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor=monitor, factor=0.5, patience=8, min_lr=1e-4
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=str(run_dir / "best_model.keras"),
                            monitor=monitor, save_best_only=True,
                        ),
                        epoch_tracker,
                    ]
                    if use_validation:
                        callbacks.append(tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=args.patience,
                            restore_best_weights=True,
                        ))
                    model.callbacks = callbacks

                    started = time.perf_counter()
                    model.fit(
                        xtrain=x_fit, ytrain=y_fit, xval=x_val, yval=y_val,
                        plot_test=use_validation,
                    )
                    training_seconds = time.perf_counter() - started
                    _, accuracy, _ = model.predict(xtest=x_test_run, ytest=y_test)
                    writer.writerow({
                        "dataset": dataset, "condition": condition, "seed": seed,
                        "accuracy": accuracy, "training_seconds": training_seconds,
                        "epochs_run": epoch_tracker.epochs_seen,
                        "best_epoch": epoch_tracker.best_epoch,
                    })
                    results_file.flush()
                    print(f"{dataset} | {condition} | seed={seed} | accuracy={accuracy:.4f} | epochs={epoch_tracker.best_epoch}")


if __name__ == "__main__":
    main()
