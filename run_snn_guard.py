"""
Restoring self-tolerance after spiking compilation.

run_snn_pooled.py found that on the headline protocol (pooled NSL-KDD, IVDA FPR 0.21%)
time-to-first-spike quantisation inflates the false positive rate several-fold at every T
that still saves energy: 1.80% at T = 16, 0.67% at T = 32. Boundary-aware construction puts
detectors immediately next to the self region, so normal traffic sits just outside detector
edges where a small quantisation shift pushes it inside.

Negative selection's defining invariant is self-tolerance: no detector may cover a training
self sample. The exact detectors satisfy it; their quantised images do not. Two ways of
re-imposing it on the compiled network are tested:

  self_tolerant   Per detector, shrink the radius until it covers no QUANTISED training self
                  point:  r'_k = min(r_k, min_s ||q(x_s) - c_k||) * (1 - 1e-9).
                  Uses self data only, no attack labels, no tuning - it is the negative
                  selection rule re-applied in the quantised domain.
  certified       Shrink every radius by the worst-case quantisation displacement
                  eps = sqrt(sum_i s_i^2) / (2T). For the consistent encoding the compiled
                  decision is exact for the quantised point q(x), and ||x - q(x)|| <= eps, so
                  "SNN fires" implies "exact detector fires": FPR_snn <= FPR_exact is
                  guaranteed. The price is detection rate.

Both are compared with the unguarded consistent encoding on the identical ten folds and seeds
as run_ablation_nslkdd.py. Energy is unaffected - only thresholds change.

Detector sets are cached to results/snn_detectors_pooled.npz for later experiments.
Writes results/snn_guard_folds.csv and results/snn_guard_summary.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from run_snn_encodings import decide, encode, E_AC, E_MAC

HP = dict(alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
          max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6)
T_GRID = [8, 16, 32, 64]
VARIANTS = {
    "V3_ivda": dict(use_statistical_term=True, use_boundary_aware=True),
    "V0_classical": dict(use_statistical_term=False, use_boundary_aware=False),
}
CACHE = Path("results/snn_detectors_pooled.npz")


def quantise(Xc, scale, T):
    """Coordinates exactly as the consistent TTFS encoder emits them."""
    V = np.clip(Xc / scale, -1.0, 1.0)
    return np.sign(V) * np.round(np.abs(V) * T) / T * scale


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    p = X.shape[1]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y))

    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    rows = []
    for fold_i, (tr, te) in enumerate(folds):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        S, A = X_tr[y_tr == "normal"], X_tr[y_tr != "normal"]
        attack, normal = y_te != "normal", y_te == "normal"
        mu = S.mean(axis=0)
        scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
        nscale = float((scale ** 2).sum())
        Sc = S - mu

        for name, flags in VARIANTS.items():
            kc, kr = f"{name}_f{fold_i}_c", f"{name}_f{fold_i}_r"
            if kc in cache:
                centers, radii = cache[kc], cache[kr]
            else:
                det = VDetectorSet(rng=np.random.default_rng(fold_i), **flags, **HP)
                det.fit(S, lo, hi, non_self_samples=A)
                centers, radii = det.centers_, det.radii_
                cache[kc], cache[kr] = centers, radii
                np.savez(CACHE, **cache)
            n = len(centers)
            C = centers - mu
            # exact decision, identical to VDetectorSet.predict_nonself
            Xc = X_te - mu
            exact = decide(Xc, (Xc ** 2).sum(axis=1), C, (C ** 2).sum(axis=1) - radii ** 2)
            dr_x, fpr_x = float(exact[attack].mean()), float(exact[normal].mean())
            dense_pJ = n * (p + 1) * E_MAC

            for T in T_GRID:
                Xh, Nh, spk = encode(X_te, mu, scale, nscale, "ttfs_consistent", T, None)
                synops = spk.mean() * n
                e_tot = (synops + n * T) * E_AC

                # self-tolerant radii: nearest quantised training self point per detector
                d_self, _ = cKDTree(quantise(Sc, scale, T)).query(C, k=1)
                r_tol = np.minimum(radii, d_self) * (1 - 1e-9)
                eps = np.sqrt(nscale) / (2 * T)
                r_cert = np.maximum(radii - eps, 0.0)

                for guard, r in [("none", radii), ("self_tolerant", r_tol), ("certified", r_cert)]:
                    pred = decide(Xh, Nh, C, (C ** 2).sum(axis=1) - r ** 2)
                    rows.append(dict(
                        fold=fold_i, variant=name, n_det=n, T=T, guard=guard,
                        agree=float((pred == exact).mean()),
                        DR=float(pred[attack].mean()), FPR=float(pred[normal].mean()),
                        DR_exact=dr_x, FPR_exact=fpr_x,
                        shrunk=int((r < radii).sum()),
                        mean_shrink=float(((radii - r) / np.maximum(radii, 1e-12)).mean()),
                        spikes=float(spk.mean()), E_total_pJ=e_tot, E_dense_pJ=dense_pJ))
            sel = [r for r in rows if r["fold"] == fold_i and r["variant"] == name and r["T"] == 16]
            print(f"fold {fold_i} {name:12s} exact DR={dr_x:.4f} FPR={fpr_x:.4f} | T=16 " +
                  "  ".join(f"{r['guard']}: DR={r['DR']:.4f} FPR={r['FPR']:.4f}" for r in sel),
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/snn_guard_folds.csv", index=False)
    summ = (df.groupby(["variant", "guard", "T"])
              .agg(n_det=("n_det", "mean"), agree=("agree", "mean"),
                   DR=("DR", "mean"), DR_sd=("DR", "std"), DR_exact=("DR_exact", "mean"),
                   FPR=("FPR", "mean"), FPR_sd=("FPR", "std"), FPR_exact=("FPR_exact", "mean"),
                   FPR_max=("FPR", "max"), shrunk=("shrunk", "mean"),
                   mean_shrink=("mean_shrink", "mean"),
                   E_total_pJ=("E_total_pJ", "mean"), E_dense_pJ=("E_dense_pJ", "mean"))
              .reset_index())
    summ["ratio_total"] = summ["E_total_pJ"] / summ["E_dense_pJ"]
    summ.to_csv("results/snn_guard_summary.csv", index=False)
    print("\n=== pooled ten-fold, consistent TTFS, mean over folds ===")
    with pd.option_context("display.width", 260, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print(summ[["variant", "guard", "T", "agree", "DR", "DR_exact", "FPR", "FPR_max",
                    "FPR_exact", "shrunk", "mean_shrink", "E_total_pJ",
                    "ratio_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
