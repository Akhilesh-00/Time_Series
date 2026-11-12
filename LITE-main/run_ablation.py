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


@tf.keras.utils.register_keras_serializable(package="ablation")
class TimeSeriesAugment(tf.keras.layers.Layer):
    """Training-only augmentation: per-sample amplitude scaling, Gaussian jitter and a
    random circular time shift. Identity at inference, so test predictions are unaffected."""

    def __init__(self, length, scale=0.1, jitter=0.03, max_shift=0.1, **kwargs):
        super().__init__(**kwargs)
        self.length, self.scale, self.jitter, self.max_shift = length, scale, jitter, max_shift

    def call(self, x, training=None):
        if not training:
            return x
        batch = tf.shape(x)[0]
        if self.scale > 0:
            x = x * tf.random.normal([batch, 1, 1], 1.0, self.scale)
        if self.jitter > 0:
            x = x + tf.random.normal(tf.shape(x), 0.0, self.jitter)
        max_steps = int(self.max_shift * self.length)
        if max_steps > 0:
            shifts = tf.random.uniform([batch], -max_steps, max_steps + 1, dtype=tf.int32)
            idx = tf.math.floormod(tf.range(self.length)[None, :] - shifts[:, None], self.length)
            x = tf.gather(x, idx, batch_dims=1)
        return x

    def get_config(self):
        return {**super().get_config(), "length": self.length, "scale": self.scale,
                "jitter": self.jitter, "max_shift": self.max_shift}


def add_augmentation(lite, length, channels, args):
    """Put a TimeSeriesAugment layer in front of the LITE network and recompile as LITE does."""
    inputs = tf.keras.layers.Input((length, channels))
    augmented = TimeSeriesAugment(length, args.aug_scale, args.aug_jitter, args.aug_shift)(inputs)
    lite.model = tf.keras.models.Model(inputs, lite.model(augmented))
    lite.model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])


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
        choices=["original", "validation", "first_difference", "validation_and_difference",
                 "validation_refit", "augment"],
        default=["original", "validation"],
    )
    parser.add_argument("--aug-scale", type=float, default=0.1, help="sd of per-sample amplitude scaling")
    parser.add_argument("--aug-jitter", type=float, default=0.03, help="sd of additive noise (data is z-normalised)")
    parser.add_argument("--aug-shift", type=float, default=0.1, help="max circular shift as a fraction of length")
    parser.add_argument("--lr-patience", type=int, default=8,
                        help="ReduceLROnPlateau patience (LITE paper uses 50)")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if not 0 < args.val_split < 1:
        parser.error("--val-split must be between 0 and 1")
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results_per_run.csv"
    fields = ["dataset", "condition", "seed", "accuracy", "training_seconds", "epochs_run", "best_epoch"]
    # validation_refit: pick the epoch count with early stopping on a validation split,
    # then retrain from scratch on the full training set for that many epochs.
    all_conditions = [
        ("original", False, False),
        ("validation", True, False),
        ("first_difference", False, True),
        ("validation_and_difference", True, True),
        ("validation_refit", True, False),
        ("augment", False, False),
    ]

    # Append so rerunning into an existing output dir never wipes earlier results.
    # Runs already in the CSV are skipped, so an interrupted job can simply be restarted.
    new_file = not results_path.exists()
    done = set()
    if not new_file:
        with results_path.open(newline="") as existing:
            done = {(r["dataset"], r["condition"], int(r["seed"])) for r in csv.DictReader(existing)}
    with results_path.open("a", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fields)
        if new_file:
            writer.writeheader()
        results_file.flush()

        for dataset in args.datasets:
            x_train, y_train, x_test, y_test = load_data(file_name=dataset)
            for seed in args.seeds:
                for condition, use_validation, use_difference in all_conditions:
                    if condition not in args.conditions:
                        continue
                    if (dataset, condition, seed) in done:
                        print(f"{dataset} | {condition} | seed={seed} | already done, skipping", flush=True)
                        continue
                    print(f"{time.strftime('%H:%M:%S')} start {dataset} | {condition} | seed={seed}", flush=True)
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
                    if condition == "augment":
                        add_augmentation(model, x_fit.shape[1], x_fit.shape[2], args)
                    callbacks = [
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor=monitor, factor=0.5, patience=args.lr_patience, min_lr=1e-4
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
                    epochs_run = epoch_tracker.epochs_seen
                    best_epoch = epoch_tracker.best_epoch

                    if condition == "validation_refit":
                        tf.keras.utils.set_random_seed(seed)
                        refit_dir = run_dir / "refit"
                        refit_dir.mkdir(parents=True, exist_ok=True)
                        model = LITE(
                            output_directory=f"{refit_dir}/", length_TS=x_train.shape[1],
                            n_channels=x_train.shape[2], n_classes=len(np.unique(y_train)),
                            n_epochs=best_epoch, verbose=args.verbose,
                        )
                        model.callbacks = [
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="loss", factor=0.5, patience=args.lr_patience, min_lr=1e-4
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath=str(refit_dir / "best_model.keras"),
                                monitor="loss", save_best_only=True,
                            ),
                        ]
                        model.fit(xtrain=x_train, ytrain=y_train, plot_test=False)
                        epochs_run += best_epoch

                    training_seconds = time.perf_counter() - started
                    _, accuracy, _ = model.predict(xtest=x_test_run, ytest=y_test)
                    writer.writerow({
                        "dataset": dataset, "condition": condition, "seed": seed,
                        "accuracy": accuracy, "training_seconds": training_seconds,
                        "epochs_run": epochs_run,
                        "best_epoch": best_epoch,
                    })
                    results_file.flush()
                    print(f"{dataset} | {condition} | seed={seed} | accuracy={accuracy:.4f} | epochs={epoch_tracker.best_epoch}")


if __name__ == "__main__":
    main()
