"""
Diagnostic for the canonical-split reversal found by run_canonical_split.py.

On the canonical KDDTrain+/KDDTest+ split the full IVDA (V3) detects 17
detection-rate points FEWER attacks than the classical V-detector (V0) it was
built to improve on, while keeping a lower false positive rate and two orders
of magnitude fewer detectors. The pooled ten-fold protocol shows the opposite.

The hypothesis: boundary-aware construction consults labelled non-self samples
(coverage probes and boundary-self points), so it specialises the detector set
to the attack distribution observed at training time. KDDTest+ contains 17
attack types absent from KDDTrain+. If the hypothesis holds, V3's deficit
should be concentrated on those unseen types, and V3 should be competitive with
V0 on attack types it saw during training.

This script splits the canonical test set's attacks into SEEN (attack label
present in KDDTrain+) and UNSEEN (absent) and reports detection rate on each,
for V0 and V3, over the same seeds.

Writes results/nslkdd_canonical_unseen.csv.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet

PCA_DIM = 15
SEEDS = [0, 1, 2, 3, 4]

DETECTOR_HP = dict(
    alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
    max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6,
)

VARIANTS = {
    "V0_classical":   dict(use_statistical_term=False, use_boundary_aware=False),
    "V2_boundary":    dict(use_statistical_term=False, use_boundary_aware=True),
    "V3_full_binary": dict(use_statistical_term=True,  use_boundary_aware=True),
}


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    X_tr, y_tr, X_te, y_te, _p, _s = build_nslkdd_features(
        train, test, pca_dim=PCA_DIM, random_state=0)
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)

    train_labels = set(train["label"].unique())
    te_label = test["label"].values

    is_attack = y_te != "normal"
    is_normal = ~is_attack
    seen_attack = is_attack & np.isin(te_label, list(train_labels))
    unseen_attack = is_attack & ~np.isin(te_label, list(train_labels))

    print(f"canonical test set: {is_normal.sum()} normal, "
          f"{seen_attack.sum()} attacks of types seen in training, "
          f"{unseen_attack.sum()} attacks of types NOT seen in training")

    self_tr = X_tr[y_tr == "normal"]
    nonself_tr = X_tr[y_tr != "normal"]

    rows = []
    for seed in SEEDS:
        for name, flags in VARIANTS.items():
            t0 = time.time()
            det = VDetectorSet(rng=np.random.default_rng(seed), **flags, **DETECTOR_HP)
            det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
            fit_s = time.time() - t0

            pred = det.predict_nonself(X_te)
            dr_seen = float(pred[seen_attack].mean())
            dr_unseen = float(pred[unseen_attack].mean())
            dr_all = float(pred[is_attack].mean())
            fpr = float(pred[is_normal].mean())

            rows.append(dict(variant=name, seed=seed,
                             DR_seen=dr_seen, DR_unseen=dr_unseen,
                             DR_all=dr_all, FPR=fpr,
                             n_detectors=det.n_detectors, fit_s=fit_s))
            print(f"  seed {seed} {name}: DR_seen={dr_seen:.4f} "
                  f"DR_unseen={dr_unseen:.4f} DR_all={dr_all:.4f} "
                  f"FPR={fpr:.4f} n_det={det.n_detectors}")

    df = pd.DataFrame(rows)
    df.to_csv("results/nslkdd_canonical_unseen.csv", index=False)

    summary = (df.groupby("variant")
                 .agg(DR_seen_mean=("DR_seen", "mean"), DR_seen_sd=("DR_seen", "std"),
                      DR_unseen_mean=("DR_unseen", "mean"), DR_unseen_sd=("DR_unseen", "std"),
                      DR_all_mean=("DR_all", "mean"),
                      FPR_mean=("FPR", "mean"),
                      n_det_mean=("n_detectors", "mean"))
                 .reset_index())
    print("\n=== seen vs unseen attack types (mean over "
          f"{len(SEEDS)} seeds) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary.to_string(index=False))
    summary.to_csv("results/nslkdd_canonical_unseen_summary.csv", index=False)
    print("\nwrote results/nslkdd_canonical_unseen.csv "
          "and results/nslkdd_canonical_unseen_summary.csv")


if __name__ == "__main__":
    main()
