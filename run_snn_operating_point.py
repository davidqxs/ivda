"""
Is the spiking network's detection-rate loss caused by quantisation, or by the operating point?

run_snn_guard.py restores self-tolerance after compilation by shrinking each detector until it
covers no quantised training self point. On the pooled ten folds that brings IVDA's FPR back to
0.15% at T = 32 (exact detector: 0.21%), but the detection rate falls from 92.35% to 90.54%.

Shrinking a detector moves it along the detection/false-alarm trade-off whether or not anything
is quantised. So the 1.81-point loss could be entirely an operating-point shift, with the
spiking network contributing nothing further - or quantisation could cost detection on top.
The two are separated here:

  exact_curve    The exact, unquantised V-detector with every radius scaled by alpha in
                 [0.80, 1.00]: the trade-off curve available without any spiking network.
  exact_tolerant The exact, unquantised V-detector with the self-tolerant radii of
                 run_snn_guard.py - same thresholds as the SNN, no input quantisation.
  snn_tolerant   The spiking network itself, consistent TTFS, self-tolerant radii.

If snn_tolerant lies on the exact curve, quantisation is free once self-tolerance is restored,
and the SNN's cost is only a choice of operating point. If it lies below the curve, the gap is
what quantisation costs.

Uses the detector sets cached by run_snn_guard.py; no detector is refitted.
Writes results/snn_operating_point.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from run_snn_encodings import decide, encode
from run_snn_guard import quantise, CACHE

ALPHAS = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.00]
T_GRID = [16, 32, 64]


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y))
    cache = dict(np.load(CACHE))

    rows = []
    for fold_i, (tr, te) in enumerate(folds):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        S = X_tr[y_tr == "normal"]
        attack, normal = y_te != "normal", y_te == "normal"
        mu = S.mean(axis=0)
        scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
        nscale = float((scale ** 2).sum())
        Sc, Xc = S - mu, X_te - mu
        Nc = (Xc ** 2).sum(axis=1)

        for name in ("V3_ivda", "V0_classical"):
            centers, radii = cache[f"{name}_f{fold_i}_c"], cache[f"{name}_f{fold_i}_r"]
            C = centers - mu
            cc = (C ** 2).sum(axis=1)

            for a in ALPHAS:
                pred = decide(Xc, Nc, C, cc - (a * radii) ** 2)
                rows.append(dict(fold=fold_i, variant=name, kind="exact_curve", param=a,
                                 DR=float(pred[attack].mean()), FPR=float(pred[normal].mean())))

            for T in T_GRID:
                d_self, _ = cKDTree(quantise(Sc, scale, T)).query(C, k=1)
                r_tol = np.minimum(radii, d_self) * (1 - 1e-9)
                th = cc - r_tol ** 2
                pred = decide(Xc, Nc, C, th)
                rows.append(dict(fold=fold_i, variant=name, kind="exact_tolerant", param=T,
                                 DR=float(pred[attack].mean()), FPR=float(pred[normal].mean())))
                Xh, Nh, _ = encode(X_te, mu, scale, nscale, "ttfs_consistent", T, None)
                pred = decide(Xh, Nh, C, th)
                rows.append(dict(fold=fold_i, variant=name, kind="snn_tolerant", param=T,
                                 DR=float(pred[attack].mean()), FPR=float(pred[normal].mean())))
        print(f"fold {fold_i} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/snn_operating_point.csv", index=False)
    summ = (df.groupby(["variant", "kind", "param"])
              .agg(DR=("DR", "mean"), FPR=("FPR", "mean")).reset_index())
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for v in ("V3_ivda", "V0_classical"):
            print(f"\n=== {v} (mean over ten folds) ===")
            print(summ[summ.variant == v].to_string(index=False))


if __name__ == "__main__":
    main()
