"""
Canonical KDDTrain+/KDDTest+ evaluation.

Every other NSL-KDD result in this project pools KDDTrain+ and KDDTest+ and
partitions the pool into ten stratified folds (run_ablation_nslkdd.py). That
protocol gives stable folds and keeps U2R present in every fold, but it removes
the deliberate train/test distribution shift the canonical NSL-KDD split was
built to impose: KDDTest+ contains attack types absent from KDDTrain+. Numbers
from the pooled protocol are therefore optimistic and not comparable with the
published literature, as the paper's Discussion states.

This script closes that gap. It runs the classical V-detector (V0), the full
binary IVDA (V3) and the full five-class IVDA (V4) on the canonical split, with
preprocessing fitted on KDDTrain+ only and applied unchanged to KDDTest+ - so
there is no leakage of any kind, unlike the pooled protocol where the one-hot,
min-max and PCA transforms are fitted once over both files.

Because a single fixed split has no fold-to-fold variance to report, the run is
replicated over several random seeds. That also separates the stochasticity of
candidate sampling from partition variance, which the pooled ten-fold results
(single seed) cannot do.

Writes results/nslkdd_canonical_runs.csv (one row per seed x variant) and
results/nslkdd_canonical_summary.csv (mean +/- SD over seeds).
"""
import time
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from ivda.multiclass import MulticlassIVDA

PCA_DIM = 15
SEEDS = [0, 1, 2, 3, 4]

# Identical to run_ablation_nslkdd.py: the canonical-split result must not be
# confounded by a hyperparameter change.
DETECTOR_HP = dict(
    alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
    max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6,
)

BINARY_VARIANTS = {
    "V0_classical":   dict(use_statistical_term=False, use_boundary_aware=False),
    "V3_full_binary": dict(use_statistical_term=True,  use_boundary_aware=True),
}


def confusion_metrics(y_true_nonself, y_pred_nonself):
    tp = np.sum(y_pred_nonself & y_true_nonself)
    fn = np.sum(~y_pred_nonself & y_true_nonself)
    fp = np.sum(y_pred_nonself & ~y_true_nonself)
    tn = np.sum(~y_pred_nonself & ~y_true_nonself)
    return dict(
        DR=tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        FPR=fp / (fp + tn) if (fp + tn) > 0 else np.nan,
        Acc=(tp + tn) / (tp + tn + fp + fn),
        TP=int(tp), FN=int(fn), FP=int(fp), TN=int(tn),
    )


def main():
    print("Loading NSL-KDD (canonical KDDTrain+/KDDTest+ split)...")
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")

    # Preprocessing fitted on the training file only, then applied to the test
    # file - the point of this script.
    X_tr, y_tr, X_te, y_te, _pipe, _scaler = build_nslkdd_features(
        train, test, pca_dim=PCA_DIM, random_state=0
    )
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)

    print(f"train N={len(X_tr)}, test N={len(X_te)}, dims={X_tr.shape[1]}")
    tr_u, tr_c = np.unique(y_tr, return_counts=True)
    te_u, te_c = np.unique(y_te, return_counts=True)
    print(f"  train classes: {dict(zip(tr_u, tr_c.tolist()))}")
    print(f"  test  classes: {dict(zip(te_u, te_c.tolist()))}")
    # Attack labels present in the test file but not in the training file: the
    # distribution shift the canonical protocol exists to impose.
    unseen = sorted(set(test["label"].unique()) - set(train["label"].unique()))
    print(f"  attack types in KDDTest+ absent from KDDTrain+: {len(unseen)} -> {unseen}")

    self_tr = X_tr[y_tr == "normal"]
    nonself_tr = X_tr[y_tr != "normal"]
    y_te_nonself = y_te != "normal"

    rows = []

    for seed in SEEDS:
        for name, flags in BINARY_VARIANTS.items():
            t0 = time.time()
            det = VDetectorSet(rng=np.random.default_rng(seed), **flags, **DETECTOR_HP)
            det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
            train_time = time.time() - t0

            t0 = time.time()
            pred = det.predict_nonself(X_te)
            infer_time = (time.time() - t0) / len(X_te)

            m = confusion_metrics(y_te_nonself, pred)
            rows.append(dict(variant=name, seed=seed, **m,
                             n_detectors=det.n_detectors,
                             train_time_s=train_time,
                             infer_time_per_sample_s=infer_time))
            print(f"  seed {seed} {name}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} "
                  f"Acc={m['Acc']:.4f} n_det={det.n_detectors} t={train_time:.1f}s")

        # V4: full five-class IVDA on the same canonical split.
        t0 = time.time()
        mc = MulticlassIVDA(detector_kwargs=dict(
            use_statistical_term=True, use_boundary_aware=True, **DETECTOR_HP))
        mc.fit(X_tr, y_tr, lo, hi, rng=np.random.default_rng(seed))
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = mc.predict(X_te)
        infer_time = (time.time() - t0) / len(X_te)
        acc = float(np.mean(y_pred == y_te))

        rows.append(dict(variant="V4_full_multiclass", seed=seed,
                         DR=np.nan, FPR=np.nan, Acc=acc,
                         TP=-1, FN=-1, FP=-1, TN=-1,
                         n_detectors=mc.n_detectors_total,
                         train_time_s=train_time,
                         infer_time_per_sample_s=infer_time))
        print(f"  seed {seed} V4_full_multiclass: Acc={acc:.4f} "
              f"n_det={mc.n_detectors_total} t={train_time:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv("results/nslkdd_canonical_runs.csv", index=False)

    summary = (df.groupby("variant")
                 .agg(DR_mean=("DR", "mean"), DR_sd=("DR", "std"),
                      FPR_mean=("FPR", "mean"), FPR_sd=("FPR", "std"),
                      Acc_mean=("Acc", "mean"), Acc_sd=("Acc", "std"),
                      n_det_mean=("n_detectors", "mean"), n_det_sd=("n_detectors", "std"),
                      train_time_mean=("train_time_s", "mean"),
                      infer_time_mean=("infer_time_per_sample_s", "mean"),
                      n_seeds=("seed", "count"))
                 .reset_index())
    summary.to_csv("results/nslkdd_canonical_summary.csv", index=False)

    print("\n=== canonical split summary (mean +/- SD over "
          f"{len(SEEDS)} seeds) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.to_string(index=False))
    print("\nwrote results/nslkdd_canonical_runs.csv and "
          "results/nslkdd_canonical_summary.csv")


if __name__ == "__main__":
    main()
