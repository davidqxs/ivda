"""
The four-encoding comparison of run_snn_encodings.py, on the headline protocol.

run_snn_encodings.py compared rate, centred rate, centred TTFS and consistent TTFS coding on
the canonical split with a single seed. The paper reports spiking results on the pooled
ten folds, so the encoding study is repeated there with the detector sets cached by
run_snn_guard.py (no refitting). Unguarded radii throughout: this table isolates the encoding;
the self-tolerance guard is reported separately (run_snn_guard.py, run_snn_rigor.py).

Writes results/snn_encodings_pooled.csv and results/snn_encodings_pooled_summary.csv.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from run_snn_encodings import decide, encode, ENCODINGS, E_AC, E_MAC
from run_snn_guard import CACHE

T_GRID = [4, 8, 16, 32, 64, 128]


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    p = X.shape[1]
    cache = dict(np.load(CACHE))

    rows = []
    for fold_i, (tr, te) in enumerate(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y)):
        S = X[tr][y[tr] == "normal"]
        X_te = X[te]
        attack, normal = y[te] != "normal", y[te] == "normal"
        mu_c = S.mean(axis=0)
        for name in ("V3_ivda", "V0_classical"):
            centers, radii = cache[f"{name}_f{fold_i}_c"], cache[f"{name}_f{fold_i}_r"]
            n = len(centers)
            dense = n * (p + 1) * E_MAC
            Xc0 = X_te - mu_c
            C0 = centers - mu_c
            exact = decide(Xc0, (Xc0 ** 2).sum(axis=1), C0, (C0 ** 2).sum(axis=1) - radii ** 2)
            fpr_x = float(exact[normal].mean())
            for enc in ENCODINGS:
                mu = mu_c if enc != "rate" else np.zeros(p)
                C = centers - mu
                theta = (C ** 2).sum(axis=1) - radii ** 2
                scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
                nscale = float((scale ** 2).sum())
                rng = np.random.default_rng(1000 + fold_i)
                for T in T_GRID:
                    Xh, Nh, spk = encode(X_te, mu, scale, nscale, enc, T, rng)
                    pred = decide(Xh, Nh, C, theta)
                    synops = spk.mean() * n
                    rows.append(dict(fold=fold_i, variant=name, n_det=n, encoding=enc, T=T,
                                     agree=float((pred == exact).mean()),
                                     DR=float(pred[attack].mean()), FPR=float(pred[normal].mean()),
                                     FPR_exact=fpr_x, spikes=float(spk.mean()),
                                     E_total_pJ=(synops + n * T) * E_AC,
                                     E_synops_pJ=synops * E_AC, E_dense_pJ=dense))
        print(f"fold {fold_i} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/snn_encodings_pooled.csv", index=False)
    s = (df.groupby(["variant", "encoding", "T"])
           .agg(agree=("agree", "mean"), DR=("DR", "mean"), FPR=("FPR", "mean"),
                FPR_exact=("FPR_exact", "mean"), spikes=("spikes", "mean"),
                E_total_pJ=("E_total_pJ", "mean"), E_synops_pJ=("E_synops_pJ", "mean"),
                E_dense_pJ=("E_dense_pJ", "mean"))
           .reset_index())
    s["ratio_total"] = s["E_total_pJ"] / s["E_dense_pJ"]
    s["ratio_synops"] = s["E_synops_pJ"] / s["E_dense_pJ"]
    s.to_csv("results/snn_encodings_pooled_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
