"""
Does a hybrid detector set repair the canonical-split failure?

`run_canonical_unseen.py` established the failure mode: boundary-aware
construction ties detector placement to the observed non-self sample, so the
resulting 81-detector set fits known attack types tightly (DR 70.7%) and covers
novel ones barely better than chance (DR 39.7%), where the classical
V-detector's 1500 blindly placed detectors reach 80.1% on novel types.

The two mechanisms fail in opposite directions, which suggests combining them:
keep the boundary-aware set for its low false positive rate, then fill part of
the detector budget with distribution-blind classical detectors to restore
coverage of the non-self region the boundary-aware rule never looks at. A
sample is flagged non-self if ANY detector in the union matches it, so the
union's detection rate is monotone in the fill count while its false positive
rate also rises - the question is the exchange rate, and whether any mixing
ratio dominates both pure variants.

For each seed one classical set of 1500 detectors is generated and the sweep
takes nested prefixes of it (classical generation places candidates i.i.d., so
a prefix is itself a valid classical run of that size). Matching uses the exact
rule of VDetectorSet.match_score, computed in chunks.

Writes results/nslkdd_canonical_hybrid.csv and a summary CSV.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet

PCA_DIM = 15
SEEDS = [0, 1, 2, 3, 4]
FILL_COUNTS = [0, 50, 100, 200, 400, 800, 1500]
CLASSICAL_POOL = 1500

DETECTOR_HP = dict(
    alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
    max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6,
)


def exact_match_any(X, centers, radii, chunk=4000):
    """Exact 'matched by at least one detector' mask - same rule as
    VDetectorSet.match_score, chunked so the distance matrix stays small."""
    n = len(X)
    if centers is None or len(centers) == 0:
        return np.zeros(n, dtype=bool)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    out = np.zeros(n, dtype=bool)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = cdist(X[s:e], centers)
        out[s:e] = (d < radii[None, :]).any(axis=1)
    return out


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

    self_tr = X_tr[y_tr == "normal"]
    nonself_tr = X_tr[y_tr != "normal"]

    print(f"test: {is_normal.sum()} normal, {seen_attack.sum()} seen-type "
          f"attacks, {unseen_attack.sum()} unseen-type attacks")

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        v3 = VDetectorSet(rng=np.random.default_rng(seed),
                          use_statistical_term=True, use_boundary_aware=True,
                          **DETECTOR_HP)
        v3.fit(self_tr, lo, hi, non_self_samples=nonself_tr)

        v0 = VDetectorSet(rng=np.random.default_rng(seed + 10_000),
                          use_statistical_term=False, use_boundary_aware=False,
                          **{**DETECTOR_HP, "max_detectors": CLASSICAL_POOL})
        v0.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
        fit_s = time.time() - t0

        n3 = v3.n_detectors
        m3 = exact_match_any(X_te, v3.centers_, v3.radii_)
        print(f"seed {seed}: boundary-aware set = {n3} detectors, "
              f"classical pool = {v0.n_detectors} (fit {fit_s:.1f}s)")

        for k in FILL_COUNTS:
            mk = exact_match_any(X_te, v0.centers_[:k], v0.radii_[:k])
            pred = m3 | mk

            dr_seen = float(pred[seen_attack].mean())
            dr_unseen = float(pred[unseen_attack].mean())
            dr_all = float(pred[is_attack].mean())
            fpr = float(pred[is_normal].mean())
            acc = float(((pred & is_attack) | (~pred & is_normal)).mean())

            rows.append(dict(seed=seed, fill=k, n_boundary=n3, n_total=n3 + k,
                             DR_seen=dr_seen, DR_unseen=dr_unseen,
                             DR_all=dr_all, FPR=fpr, Acc=acc))
            print(f"    fill={k:4d} (total {n3 + k:4d}): DR_seen={dr_seen:.4f} "
                  f"DR_unseen={dr_unseen:.4f} DR_all={dr_all:.4f} "
                  f"FPR={fpr:.4f} Acc={acc:.4f}")

        # pure classical reference at the same pool size, no boundary-aware set
        m0 = exact_match_any(X_te, v0.centers_, v0.radii_)
        rows.append(dict(seed=seed, fill=-1, n_boundary=0, n_total=v0.n_detectors,
                         DR_seen=float(m0[seen_attack].mean()),
                         DR_unseen=float(m0[unseen_attack].mean()),
                         DR_all=float(m0[is_attack].mean()),
                         FPR=float(m0[is_normal].mean()),
                         Acc=float(((m0 & is_attack) | (~m0 & is_normal)).mean())))

    df = pd.DataFrame(rows)
    df.to_csv("results/nslkdd_canonical_hybrid.csv", index=False)

    summary = (df.groupby("fill")
                 .agg(n_total=("n_total", "mean"),
                      DR_seen=("DR_seen", "mean"), DR_seen_sd=("DR_seen", "std"),
                      DR_unseen=("DR_unseen", "mean"), DR_unseen_sd=("DR_unseen", "std"),
                      DR_all=("DR_all", "mean"),
                      FPR=("FPR", "mean"), FPR_sd=("FPR", "std"),
                      Acc=("Acc", "mean"), Acc_sd=("Acc", "std"))
                 .reset_index()
                 .sort_values("fill"))
    summary.to_csv("results/nslkdd_canonical_hybrid_summary.csv", index=False)

    print("\n=== hybrid sweep (mean over 5 seeds; fill=-1 is pure classical) ===")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summary.to_string(index=False))
    print("\nwrote results/nslkdd_canonical_hybrid.csv and "
          "results/nslkdd_canonical_hybrid_summary.csv")


if __name__ == "__main__":
    main()
