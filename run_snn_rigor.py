"""
Robustness checks for the spiking results.

Two claims rest so far on NSL-KDD alone and on a coarse grid:

 (1) Spiking compilation with consistent TTFS coding and self-tolerance restoration keeps the
     V-detector on its own detection/false-alarm frontier at a fraction of the dense energy.
     -> replicated here on UNSW-NB15, the paper's second dataset.

 (2) Shrinking each detector just enough to restore self-tolerance gives a better operating
     point than shrinking all radii uniformly. The earlier evidence sampled alpha at 0.99 and
     1.00 only.
     -> tested here on both datasets with alpha in [0.95, 1.00] at step 0.0005. The comparison
     is deliberately biased TOWARDS uniform shrinking: for every fold/seed, uniform scaling
     picks, after the fact and on the test data, the alpha with the highest DR among those
     whose FPR does not exceed the self-tolerant detector's. Self-tolerant shrinking is not
     tuned at all. If it still wins, the claim holds; if not, it is withdrawn.

NSL-KDD: the pooled ten folds, using the detector sets cached by run_snn_guard.py.
UNSW-NB15: the paper's split (stratified 80/20 on attack_cat, random_state 0), detector seeds
0-4. Seed 0 reproduces the Section 4.10 numbers.

Writes results/snn_rigor_{nslkdd,unsw}.csv, results/snn_rigor_matched.csv and a summary to
stdout.
"""
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features, load_unsw, build_unsw_features
from ivda.detector import VDetectorSet
from run_snn_encodings import decide, encode, E_AC, E_MAC
from run_snn_guard import quantise, CACHE

HP = dict(alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
          max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6)
VARIANTS = {"V3_ivda": dict(use_statistical_term=True, use_boundary_aware=True),
            "V0_classical": dict(use_statistical_term=False, use_boundary_aware=False)}
T_GRID = [8, 16, 32, 64]
ALPHAS = np.round(np.arange(0.95, 1.0 + 1e-9, 0.0005), 4)


def evaluate(unit, dataset, name, centers, radii, S, X_te, attack, normal, lo, hi):
    """All SNN and exact measurements for one detector set. Returns (rows, matched rows)."""
    p = X_te.shape[1]
    n = len(centers)
    mu = S.mean(axis=0)
    scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
    nscale = float((scale ** 2).sum())
    C = centers - mu
    cc = (C ** 2).sum(axis=1)
    Xc = X_te - mu
    Nc = (Xc ** 2).sum(axis=1)
    exact = decide(Xc, Nc, C, cc - radii ** 2)
    dr_x, fpr_x = float(exact[attack].mean()), float(exact[normal].mean())
    dense = n * (p + 1) * E_MAC
    rm = lambda pred: (float(pred[attack].mean()), float(pred[normal].mean()))

    # uniform-scaling curve of the exact detector
    curve = np.array([rm(decide(Xc, Nc, C, cc - (a * radii) ** 2)) for a in ALPHAS])

    rows, matched = [], []
    for T in T_GRID:
        Xh, Nh, spk = encode(X_te, mu, scale, nscale, "ttfs_consistent", T, None)
        synops = spk.mean() * n
        e_tot = (synops + n * T) * E_AC
        d_self, _ = cKDTree(quantise(S - mu, scale, T)).query(C, k=1)
        r_tol = np.minimum(radii, d_self) * (1 - 1e-9)
        r_cert = np.maximum(radii - np.sqrt(nscale) / (2 * T), 0.0)

        exact_tol = rm(decide(Xc, Nc, C, cc - r_tol ** 2))
        for guard, r in [("none", radii), ("self_tolerant", r_tol), ("certified", r_cert)]:
            pred = decide(Xh, Nh, C, cc - r ** 2)
            dr, fpr = rm(pred)
            rows.append(dict(dataset=dataset, unit=unit, variant=name, n_det=n, T=T, guard=guard,
                             agree=float((pred == exact).mean()), DR=dr, FPR=fpr,
                             DR_exact=dr_x, FPR_exact=fpr_x,
                             spikes=float(spk.mean()), spikes_normal=float(spk[normal].mean()),
                             spikes_attack=float(spk[attack].mean()),
                             E_total_pJ=e_tot, E_synops_pJ=synops * E_AC, E_dense_pJ=dense))
        rows.append(dict(dataset=dataset, unit=unit, variant=name, n_det=n, T=T,
                         guard="exact_tolerant", agree=np.nan, DR=exact_tol[0], FPR=exact_tol[1],
                         DR_exact=dr_x, FPR_exact=fpr_x, spikes=np.nan, spikes_normal=np.nan,
                         spikes_attack=np.nan, E_total_pJ=np.nan, E_synops_pJ=np.nan,
                         E_dense_pJ=dense))

        # matched-FPR comparison, biased towards uniform scaling (it picks its alpha post hoc)
        dr_tol, fpr_tol = exact_tol
        ok = curve[:, 1] <= fpr_tol + 1e-12
        j = int(np.argmax(np.where(ok, curve[:, 0], -1.0)))
        matched.append(dict(dataset=dataset, unit=unit, variant=name, T=T,
                            DR_tolerant=dr_tol, FPR_tolerant=fpr_tol,
                            DR_uniform_best=float(curve[j, 0]), FPR_uniform=float(curve[j, 1]),
                            alpha_best=float(ALPHAS[j]), gain=dr_tol - float(curve[j, 0])))
    return rows, matched


def run_nslkdd():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(full, full.iloc[:1], pca_dim=15, random_state=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    cache = dict(np.load(CACHE))
    rows, matched = [], []
    for fold_i, (tr, te) in enumerate(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X, y)):
        S = X[tr][y[tr] == "normal"]
        attack, normal = y[te] != "normal", y[te] == "normal"
        for name in VARIANTS:
            r, m = evaluate(fold_i, "NSL-KDD", name, cache[f"{name}_f{fold_i}_c"],
                            cache[f"{name}_f{fold_i}_r"], S, X[te], attack, normal, lo, hi)
            rows += r
            matched += m
        print(f"NSL-KDD fold {fold_i} done", flush=True)
    return rows, matched


def run_unsw():
    df = load_unsw("data/unsw/train.csv")
    tr_df, te_df = train_test_split(df, test_size=0.2, stratify=df["attack_cat"], random_state=0)
    X_tr, y_tr, X_te, y_te, _, _ = build_unsw_features(tr_df, te_df, pca_dim=15, random_state=0)
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    S, A = X_tr[y_tr == "Normal"], X_tr[y_tr != "Normal"]
    attack, normal = y_te != "Normal", y_te == "Normal"
    rows, matched = [], []
    for seed in range(5):
        for name, flags in VARIANTS.items():
            det = VDetectorSet(rng=np.random.default_rng(seed), **flags, **HP)
            det.fit(S, lo, hi, non_self_samples=A)
            r, m = evaluate(seed, "UNSW-NB15", name, det.centers_, det.radii_, S, X_te,
                            attack, normal, lo, hi)
            rows += r
            matched += m
            ex = r[0]
            print(f"UNSW seed {seed} {name:12s} n={det.n_detectors:4d} exact DR={ex['DR_exact']:.4f} "
                  f"FPR={ex['FPR_exact']:.4f}", flush=True)
    return rows, matched


def summarise(rows, matched):
    df = pd.DataFrame(rows)
    mt = pd.DataFrame(matched)
    for ds, sub in df.groupby("dataset"):
        sub.to_csv(f"results/snn_rigor_{'nslkdd' if ds == 'NSL-KDD' else 'unsw'}.csv", index=False)
    mt.to_csv("results/snn_rigor_matched.csv", index=False)

    agg = (df.groupby(["dataset", "variant", "guard", "T"])
             .agg(n_det=("n_det", "mean"), agree=("agree", "mean"),
                  DR=("DR", "mean"), DR_sd=("DR", "std"), DR_exact=("DR_exact", "mean"),
                  FPR=("FPR", "mean"), FPR_max=("FPR", "max"), FPR_exact=("FPR_exact", "mean"),
                  spikes=("spikes", "mean"), E_total_pJ=("E_total_pJ", "mean"),
                  E_dense_pJ=("E_dense_pJ", "mean"))
             .reset_index())
    agg["ratio_total"] = agg["E_total_pJ"] / agg["E_dense_pJ"]
    agg.to_csv("results/snn_rigor_summary.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print("\n=== SNN fidelity and energy (mean over folds/seeds) ===")
        print(agg.to_string(index=False))

        print("\n=== per-detector self-tolerant vs best uniform shrink at matched FPR ===")
        print("    (uniform picks its alpha post hoc on test data; self-tolerant is untuned)")
        for (ds, v, T), g in mt.groupby(["dataset", "variant", "T"]):
            d = g["gain"].values
            wins = int((d > 0).sum()); ties = int((d == 0).sum())
            try:
                pw = wilcoxon(g["DR_tolerant"], g["DR_uniform_best"]).pvalue if np.any(d != 0) else np.nan
            except ValueError:
                pw = np.nan
            pt = ttest_rel(g["DR_tolerant"], g["DR_uniform_best"]).pvalue if np.std(d) > 0 else np.nan
            print(f"  {ds:9s} {v:12s} T={T:3d}  DR_tol={g['DR_tolerant'].mean():.4f} "
                  f"DR_unif={g['DR_uniform_best'].mean():.4f}  gain={d.mean() * 100:+.2f} pts  "
                  f"wins {wins}/{len(d)} ties {ties}  alpha*={g['alpha_best'].mean():.4f}  "
                  f"Wilcoxon p={pw:.4g}  t p={pt:.4g}")


def main():
    r1, m1 = run_nslkdd()
    r2, m2 = run_unsw()
    summarise(r1 + r2, m1 + m2)


if __name__ == "__main__":
    main()
