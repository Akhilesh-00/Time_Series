"""Evaluate ensembles of saved LITE checkpoints (average of softmax outputs), like LITETime."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.utils import load_data


def add_first_difference(x):
    return np.concatenate((x, np.diff(x, axis=1, prepend=x[:, :1, :])), axis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="<output-dir>/<dataset>")
    ap.add_argument("--conditions", nargs="+", default=["original", "validation"])
    args = ap.parse_args()

    dataset = args.root.name
    _, _, x_test, y_test = load_data(file_name=dataset)
    for condition in args.conditions:
        x = add_first_difference(x_test) if "difference" in condition else x_test
        probs = {}
        for seed_dir in sorted((args.root / condition).glob("seed_*")):
            ckpt = seed_dir / ("refit" if condition == "validation_refit" else "") / "best_model.keras"
            if ckpt.exists():
                model = tf.keras.models.load_model(ckpt, compile=False)
                probs[int(seed_dir.name[5:])] = model.predict(x, verbose=0)
                tf.keras.backend.clear_session()
        seeds = sorted(probs)
        singles = [accuracy_score(y_test, probs[s].argmax(1)) for s in seeds]
        print(f"\n{dataset} | {condition} | seeds {seeds}")
        print(f"  single models: {np.round(singles, 4).tolist()}  mean {np.mean(singles):.4f}")
        for k in range(2, len(seeds) + 1):
            accs = [accuracy_score(y_test, np.mean([probs[s] for s in combo], 0).argmax(1))
                    for combo in itertools.combinations(seeds, k)]
            print(f"  ensemble of {k}: mean over {len(accs)} combos {np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
