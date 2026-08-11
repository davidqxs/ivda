"""Reproduce a fixed-radius real-valued NSA baseline under the identical pipeline.

Motivation: Table 4 previously compared IVDA only against two general-purpose machine
learning baselines (KNN, K-means) plus one literature-quoted hybrid. No method from the
NSA / V-detector family - the actual competitive field - was in the comparison. The
fixed-radius real-valued NSA is
the natural choice: it is precisely the algorithm the V-detector was introduced to
improve on, so the comparison isolates the value of the variable-radius idea and of the
boundary-aware extension on top of it.

Fairness: identical PCA-15 preprocessing, identical ten folds (StratifiedKFold, seed 0),
identical detector budget (1500) and placement-failure limit as V0-V3, and the same
per-fold RNG seeding. Only self samples are used, as a fixed-radius NSA requires no
labelled attack data. The single free parameter - the detector radius - is chosen by a
sweep that is reported in full, and the value that maximises the baseline's own accuracy
is the one carried into Table 4, so the baseline is shown at its best.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from run_ablation_nslkdd import confusion_metrics, DETECTOR_HP, RNG_SEED, N_FOLDS, PCA_DIM

# radius grid is derived from the data (percentiles of the nearest-self distance of
# uniform candidates) so that it always spans the usable range of this feature space
RADIUS_PERCENTILES = [2, 5, 10, 20, 30, 40, 50, 60]


def build_data():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=PCA_DIM,
                                             random_state=RNG_SEED)
    lo, hi = X.min(axis=0), X.max(axis=0)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    return X, y, lo, hi, list(skf.split(X, y))


def fit_eval(self_tr, X_te, y_te_nonself, lo, hi, radius, seed):
    hp = {k: v for k, v in DETECTOR_HP.items() if k not in ("cap_radius",)}
    t0 = time.time()
    det = VDetectorSet(use_statistical_term=False, use_boundary_aware=False,
                       fixed_radius=radius, cap_radius=False,
                       rng=np.random.default_rng(seed), **hp)
    det.fit(self_tr, lo, hi, non_self_samples=None)   # one-class: no attack labels used
    train_time = time.time() - t0

    t0 = time.time()
    pred = det.predict_nonself(X_te)
    infer_time = (time.time() - t0) / len(X_te)
    m = confusion_metrics(y_te_nonself, pred)
    return m, det.n_detectors, train_time, infer_time


def main():
    print("Loading NSL-KDD...")
    X, y, lo, hi, folds = build_data()
    print(f"N={len(X)}, dims={X.shape[1]}")

    # ---- radius grid from the nearest-self-distance distribution ----
    from scipy.spatial import cKDTree
    self_all = X[y == "normal"]
    rng = np.random.default_rng(RNG_SEED)
    probe = rng.uniform(lo, hi, size=(4000, X.shape[1]))
    d_self, _ = cKDTree(self_all).query(probe, k=1)
    radii = [float(np.percentile(d_self, p)) for p in RADIUS_PERCENTILES]
    print("nearest-self distance percentiles ->",
          {p: round(r, 4) for p, r in zip(RADIUS_PERCENTILES, radii)})

    # ---- phase 1: sweep the radius on fold 0 ----
    tr_idx, te_idx = folds[0]
    self_tr = X[tr_idx][y[tr_idx] == "normal"]
    X_te, y_te_nonself = X[te_idx], y[te_idx] != "normal"

    sweep = []
    for p, r in zip(RADIUS_PERCENTILES, radii):
        m, n_det, tt, it = fit_eval(self_tr, X_te, y_te_nonself, lo, hi, r, RNG_SEED * 1000)
        sweep.append(dict(percentile=p, radius=r, **m, n_detectors=n_det, train_time_s=tt))
        print(f"  r={r:.4f} (p{p:<2d}): DR={m['DR']:.4f} FPR={m['FPR']:.4f} "
              f"Acc={m['Acc']:.4f} n_det={n_det}")
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv("results/nslkdd_fixed_radius_sweep.csv", index=False)

    best = sweep_df.loc[sweep_df["Acc"].idxmax()]
    r_best = float(best["radius"])
    print(f"best radius by accuracy on fold 0: r={r_best:.4f} (p{int(best['percentile'])})")

    # ---- phase 2: the chosen radius over all ten folds ----
    rows = []
    for i, (tr_idx, te_idx) in enumerate(folds):
        self_tr = X[tr_idx][y[tr_idx] == "normal"]
        X_te, y_te_nonself = X[te_idx], y[te_idx] != "normal"
        m, n_det, tt, it = fit_eval(self_tr, X_te, y_te_nonself, lo, hi, r_best,
                                    RNG_SEED * 1000 + i)
        rows.append(dict(fold=i, **m, n_detectors=n_det, train_time_s=tt,
                         infer_time_per_sample_s=it))
        print(f"  fold {i}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} Acc={m['Acc']:.4f} "
              f"n_det={n_det} t={tt:.1f}s")
    folds_df = pd.DataFrame(rows)
    folds_df.to_csv("results/nslkdd_fixed_radius_folds.csv", index=False)

    summary = []
    for metric in ["DR", "FPR", "Acc", "n_detectors", "train_time_s",
                   "infer_time_per_sample_s"]:
        summary.append(dict(variant="FR_NSA", metric=metric,
                            mean=folds_df[metric].mean(), sd=folds_df[metric].std(ddof=1)))
    pd.DataFrame(summary).to_csv("results/nslkdd_fixed_radius.csv", index=False)

    # ---- paired tests against V3 on the same folds ----
    abl = pd.read_csv("results/nslkdd_ablation_folds.csv")
    v3 = abl[abl["variant"] == "V3_full_binary"].sort_values("fold")
    sig = []
    for metric in ["DR", "FPR", "Acc"]:
        a = folds_df.sort_values("fold")[metric].to_numpy()
        b = v3[metric].to_numpy()
        t_stat, t_p = ttest_rel(b, a)
        try:
            w_stat, w_p = wilcoxon(b, a)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        sig.append(dict(metric=metric, comparison="V3_vs_FR_NSA",
                        mean_diff=float((b - a).mean()), t_stat=t_stat, t_p=t_p,
                        wilcoxon_stat=w_stat, wilcoxon_p=w_p))
        print(f"  {metric}: V3 - FR-NSA = {(b - a).mean():+.4f}  t_p={t_p:.2e}  w_p={w_p}")
    pd.DataFrame(sig).to_csv("results/nslkdd_fixed_radius_significance.csv", index=False)

    with open("results/nslkdd_fixed_radius_hp.json", "w") as f:
        json.dump(dict(radius=r_best, radius_percentile=int(best["percentile"]),
                       radius_grid=dict(zip(map(str, RADIUS_PERCENTILES), radii)),
                       selected_on="fold 0 accuracy", n_folds=N_FOLDS, pca_dim=PCA_DIM,
                       rng_seed=RNG_SEED,
                       max_detectors=DETECTOR_HP["max_detectors"],
                       max_consecutive_failures=DETECTOR_HP["max_consecutive_failures"]),
                  f, indent=2)
    print("Done -> results/nslkdd_fixed_radius*.csv")


if __name__ == "__main__":
    main()
