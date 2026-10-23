"""
Spiking fidelity on the paper's headline protocol: pooled NSL-KDD, the same ten folds.

run_snn_encodings.py showed on the canonical split that centred time-to-first-spike coding
reproduces the compiled V-detector: for IVDA's compact set, 99.06% agreement at T = 16 with
DR and FPR essentially unchanged, at 0.36x the dense energy under the honest accounting.

The headline results, though, are on the pooled ten-fold protocol, where IVDA's false
positive rate is 0.21%. At an FPR that low the spiking network's tolerance is much tighter: a
1% disagreement rate falling on normal traffic would multiply the false positive rate several
times over. This script checks that directly, on the identical folds and seeds as
run_ablation_nslkdd.py, so every exact-V-detector number here is a Table 3 / Table 6 number.

Writes results/snn_pooled_folds.csv and results/snn_pooled_summary.csv.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from run_snn_encodings import decide, encode, E_AC, E_MAC

HP = dict(alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
          max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6)
T_GRID = [8, 16, 32, 64]
ENCODINGS = ["ttfs_centred", "ttfs_consistent"]
VARIANTS = {
    "V3_ivda": dict(use_statistical_term=True, use_boundary_aware=True),
    "V0_classical": dict(use_statistical_term=False, use_boundary_aware=False),
}


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    p = X.shape[1]
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y))

    rows = []
    for fold_i, (tr, te) in enumerate(folds):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        S, A = X_tr[y_tr == "normal"], X_tr[y_tr != "normal"]
        attack, normal = y_te != "normal", y_te == "normal"
        mu = S.mean(axis=0)
        scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
        nscale = float((scale ** 2).sum())

        for name, flags in VARIANTS.items():
            det = VDetectorSet(rng=np.random.default_rng(fold_i), **flags, **HP)
            det.fit(S, lo, hi, non_self_samples=A)
            n = det.n_detectors
            exact = det.predict_nonself(X_te)
            C = det.centers_ - mu
            theta = (C ** 2).sum(axis=1) - det.radii_ ** 2
            dense_pJ = n * (p + 1) * E_MAC
            dr_x, fpr_x = float(exact[attack].mean()), float(exact[normal].mean())

            rng = np.random.default_rng(1)
            for enc in ENCODINGS:
                for T in T_GRID:
                    Xh, Nh, spk = encode(X_te, mu, scale, nscale, enc, T, rng)
                    pred = decide(Xh, Nh, C, theta)
                    synops = spk.mean() * n
                    rows.append(dict(
                        fold=fold_i, variant=name, n_det=n, encoding=enc, T=T,
                        agree=float((pred == exact).mean()),
                        agree_normal=float((pred[normal] == exact[normal]).mean()),
                        DR=float(pred[attack].mean()), FPR=float(pred[normal].mean()),
                        DR_exact=dr_x, FPR_exact=fpr_x,
                        spikes=float(spk.mean()), spikes_normal=float(spk[normal].mean()),
                        spikes_attack=float(spk[attack].mean()),
                        E_dense_pJ=dense_pJ, E_synops_pJ=synops * E_AC,
                        E_total_pJ=(synops + n * T) * E_AC))
            last = [r for r in rows if r["fold"] == fold_i and r["variant"] == name
                    and r["encoding"] == "ttfs_consistent" and r["T"] == 16][0]
            print(f"fold {fold_i} {name:12s} n={n:4d} exact DR={dr_x:.4f} FPR={fpr_x:.4f} | "
                  f"ttfs_consistent T=16: agree={last['agree']:.4f} DR={last['DR']:.4f} "
                  f"FPR={last['FPR']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/snn_pooled_folds.csv", index=False)
    summ = (df.groupby(["variant", "encoding", "T"])
              .agg(n_det=("n_det", "mean"), agree=("agree", "mean"), agree_min=("agree", "min"),
                   agree_normal=("agree_normal", "mean"),
                   DR=("DR", "mean"), DR_exact=("DR_exact", "mean"),
                   FPR=("FPR", "mean"), FPR_sd=("FPR", "std"), FPR_exact=("FPR_exact", "mean"),
                   spikes=("spikes", "mean"), spikes_normal=("spikes_normal", "mean"),
                   spikes_attack=("spikes_attack", "mean"),
                   E_dense_pJ=("E_dense_pJ", "mean"), E_synops_pJ=("E_synops_pJ", "mean"),
                   E_total_pJ=("E_total_pJ", "mean"))
              .reset_index())
    summ["ratio_total"] = summ["E_total_pJ"] / summ["E_dense_pJ"]
    summ["ratio_synops"] = summ["E_synops_pJ"] / summ["E_dense_pJ"]
    summ.to_csv("results/snn_pooled_summary.csv", index=False)
    print("\n=== pooled ten-fold, mean over folds ===")
    with pd.option_context("display.width", 260, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print(summ[["variant", "encoding", "T", "n_det", "agree", "agree_min", "DR", "DR_exact",
                    "FPR", "FPR_exact", "spikes", "E_total_pJ", "ratio_total",
                    "ratio_synops"]].to_string(index=False))


if __name__ == "__main__":
    main()
