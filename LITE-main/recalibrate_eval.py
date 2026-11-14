"""Batch-norm recalibration of saved LITE checkpoints, plus optional mixed ensembles.

After training, the batch-norm moving statistics are replaced by statistics computed from one
pass over the full training set (training-mode forward pass with momentum 0). No retraining and
no test data are used. Reports single-model and ensemble test accuracy before and after.

Examples:
  python recalibrate_eval.py --dataset Wine --models "results_ablation/gpu_scale_small/Wine/original/seed_*/best_model.keras"
  python recalibrate_eval.py --dataset Beef --first-difference --models "results_ablation/gpu_beef_diff/Beef/first_difference/seed_*/best_model.keras"
  python recalibrate_eval.py --dataset Wine --models ".../original/seed_*/best_model.keras" --mix ".../augment/seed_*/best_model.keras"
"""
import argparse
import glob
import itertools

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import run_ablation  # registers the augmentation layer so augmented checkpoints load
from src.utils import load_data


def recalibrate(model, x_train):
    layers = list(model._flatten_layers())
    # Augmented models: switch the augmentation off so statistics come from the real training data.
    augs = [l for l in layers if isinstance(l, run_ablation.TimeSeriesAugment)]
    saved = [(l.scale, l.jitter, l.max_shift) for l in augs]
    for l in augs:
        l.scale = l.jitter = l.max_shift = 0.0
    bns = [l for l in layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    momenta = [l.momentum for l in bns]
    for l in bns:
        l.momentum = 0.0
    model(x_train, training=True)
    for l, m in zip(bns, momenta):
        l.momentum = m
    for l, s in zip(augs, saved):
        l.scale, l.jitter, l.max_shift = s


def predict_all(pattern, x_train, x_test):
    before, after = [], []
    for path in sorted(glob.glob(pattern, recursive=True)):
        model = tf.keras.models.load_model(path, compile=False)
        before.append(model.predict(x_test, verbose=0))
        recalibrate(model, x_train)
        after.append(model.predict(x_test, verbose=0))
        tf.keras.backend.clear_session()
    return before, after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", required=True, help="glob of best_model.keras checkpoints")
    ap.add_argument("--mix", help="optional second glob: report ensembles mixing both model sets")
    ap.add_argument("--first-difference", action="store_true", help="models were trained with first-difference input")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5], help="ensemble sizes averaged over all subsets")
    args = ap.parse_args()

    x_train, _, x_test, y_test = load_data(file_name=args.dataset)
    if args.first_difference:
        x_train, x_test = run_ablation.add_first_difference(x_train), run_ablation.add_first_difference(x_test)
    acc = lambda probs: accuracy_score(y_test, np.mean(probs, 0).argmax(1))

    before, after = predict_all(args.models, x_train, x_test)
    if not before:
        raise SystemExit(f"no checkpoints match {args.models}")
    print(f"{args.dataset}: {len(before)} models")
    print("  single before:", [round(acc([p]), 4) for p in before], f"mean {np.mean([acc([p]) for p in before]):.4f}")
    print("  single after: ", [round(acc([p]), 4) for p in after], f"mean {np.mean([acc([p]) for p in after]):.4f}")
    print(f"  ensemble of all: {acc(before):.4f} -> {acc(after):.4f}")
    for k in args.k:
        if k < len(before):
            combos = list(itertools.combinations(range(len(before)), k))
            b = np.mean([acc([before[i] for i in c]) for c in combos])
            a = np.mean([acc([after[i] for i in c]) for c in combos])
            print(f"  ensembles of {k}, mean over {len(combos)} subsets: {b:.4f} -> {a:.4f}")

    if args.mix:
        mix_before, _ = predict_all(args.mix, x_train, x_test)
        print(f"  mixed ensemble ({len(before)} + {len(mix_before)} models, no recalibration): {acc(before + mix_before):.4f}")


if __name__ == "__main__":
    main()
