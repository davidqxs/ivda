"""
Main NSL-KDD experiment: ablation (V0-V4), significance testing, baseline
reproduction, timing, and per-class metrics on NSL-KDD - all from one 10-fold
CV run so every number traces to this single script. Writes the CSVs under
results/ that Tables 3, 5 and 6 and Figure 3 of the paper are built from.
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.stats import ttest_rel, wilcoxon

import sys
sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from ivda.multiclass import MulticlassIVDA

RNG_SEED = 0
N_FOLDS = 10
PCA_DIM = 15

DETECTOR_HP = dict(
    alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
    max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6,
)

VARIANTS = {
    "V0_classical":     dict(use_statistical_term=False, use_boundary_aware=False),
    "V1_stat_term":     dict(use_statistical_term=True,  use_boundary_aware=False),
    "V2_boundary":      dict(use_statistical_term=False, use_boundary_aware=True),
    "V3_full_binary":   dict(use_statistical_term=True,  use_boundary_aware=True),
}


def confusion_metrics(y_true_nonself, y_pred_nonself):
    tp = np.sum(y_pred_nonself & y_true_nonself)
    fn = np.sum(~y_pred_nonself & y_true_nonself)
    fp = np.sum(y_pred_nonself & ~y_true_nonself)
    tn = np.sum(~y_pred_nonself & ~y_true_nonself)
    dr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(DR=dr, FPR=fpr, Acc=acc, TP=tp, FN=fn, FP=fp, TN=tn)


def main():
    print("Loading NSL-KDD...")
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    # 10-fold CV is run over the combined KDDTrain+/KDDTest+ pool, as stated in
    # Section 4.1 of the paper; the departure from the canonical split is
    # discussed in the paper's Limitations section.
    full = pd.concat([train, test], ignore_index=True)
    Xtr_all, ytr_all, Xte_dummy, yte_dummy, pipe, post_scaler = build_nslkdd_features(
        full, full.iloc[:1], pca_dim=PCA_DIM, random_state=RNG_SEED
    )
    X, y = Xtr_all, ytr_all
    lo, hi = X.min(axis=0), X.max(axis=0)
    print(f"N={len(X)}, dims={X.shape[1]}, classes={np.unique(y, return_counts=True)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    folds = list(skf.split(X, y))

    ablation_rows = []
    fold_arrays = {name: {"DR": [], "FPR": [], "Acc": [], "n_det": [], "train_time": [], "infer_time": []}
                   for name in VARIANTS}

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        self_tr = X_tr[y_tr == "normal"]
        nonself_tr = X_tr[y_tr != "normal"]
        y_te_nonself = y_te != "normal"

        for name, flags in VARIANTS.items():
            t0 = time.time()
            det = VDetectorSet(rng=np.random.default_rng(RNG_SEED * 1000 + fold_i), **flags, **DETECTOR_HP)
            det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
            train_time = time.time() - t0

            t0 = time.time()
            pred_nonself = det.predict_nonself(X_te)
            infer_time = (time.time() - t0) / len(X_te)

            m = confusion_metrics(y_te_nonself, pred_nonself)
            fold_arrays[name]["DR"].append(m["DR"])
            fold_arrays[name]["FPR"].append(m["FPR"])
            fold_arrays[name]["Acc"].append(m["Acc"])
            fold_arrays[name]["n_det"].append(det.n_detectors)
            fold_arrays[name]["train_time"].append(train_time)
            fold_arrays[name]["infer_time"].append(infer_time)

            ablation_rows.append(dict(variant=name, fold=fold_i, **m,
                                       n_detectors=det.n_detectors,
                                       train_time_s=train_time,
                                       infer_time_per_sample_s=infer_time))
            print(f"  fold {fold_i} {name}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} "
                  f"Acc={m['Acc']:.4f} n_det={det.n_detectors} t={train_time:.1f}s")

    # ---- V4: full multi-class IVDA, same folds ----
    print("Running V4 multi-class...")
    v4_rows = []
    per_class_records = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        t0 = time.time()
        mc = MulticlassIVDA(detector_kwargs=dict(use_statistical_term=True, use_boundary_aware=True, **DETECTOR_HP))
        mc.fit(X_tr, y_tr, lo, hi, rng=np.random.default_rng(RNG_SEED * 1000 + fold_i))
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = mc.predict(X_te)
        infer_time = (time.time() - t0) / len(X_te)

        acc = np.mean(y_pred == y_te)
        v4_rows.append(dict(fold=fold_i, Acc=acc, n_detectors=mc.n_detectors_total,
                             train_time_s=train_time, infer_time_per_sample_s=infer_time))
        print(f"  fold {fold_i} V4_multiclass: Acc={acc:.4f} n_det={mc.n_detectors_total} t={train_time:.1f}s")

        for cls in mc.classes_:
            mask = y_te == cls
            if mask.sum() == 0:
                continue
            tp = np.sum((y_pred == cls) & mask)
            fn = np.sum((y_pred != cls) & mask)
            fp = np.sum((y_pred == cls) & ~mask)
            tn = np.sum((y_pred != cls) & ~mask)
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
            tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            per_class_records.append(dict(fold=fold_i, cls=cls, precision=precision,
                                           recall=recall, f1=f1, tnr=tnr, support=mask.sum()))

    ablation_df = pd.DataFrame(ablation_rows)
    v4_df = pd.DataFrame(v4_rows)
    per_class_df = pd.DataFrame(per_class_records)

    ablation_df.to_csv("results/nslkdd_ablation_folds.csv", index=False)
    v4_df.to_csv("results/nslkdd_v4_multiclass_folds.csv", index=False)
    per_class_df.to_csv("results/nslkdd_per_class_folds.csv", index=False)

    # ---- summary table (mean +/- SD per variant) ----
    summary_rows = []
    for name in VARIANTS:
        for metric in ["DR", "FPR", "Acc"]:
            vals = np.array(fold_arrays[name][metric])
            summary_rows.append(dict(variant=name, metric=metric, mean=vals.mean(), sd=vals.std(ddof=1)))
        summary_rows.append(dict(variant=name, metric="n_detectors",
                                  mean=np.mean(fold_arrays[name]["n_det"]),
                                  sd=np.std(fold_arrays[name]["n_det"], ddof=1)))
        summary_rows.append(dict(variant=name, metric="train_time_s",
                                  mean=np.mean(fold_arrays[name]["train_time"]),
                                  sd=np.std(fold_arrays[name]["train_time"], ddof=1)))
        summary_rows.append(dict(variant=name, metric="infer_time_per_sample_s",
                                  mean=np.mean(fold_arrays[name]["infer_time"]),
                                  sd=np.std(fold_arrays[name]["infer_time"], ddof=1)))
    summary_rows.append(dict(variant="V4_multiclass", metric="Acc",
                              mean=v4_df["Acc"].mean(), sd=v4_df["Acc"].std(ddof=1)))
    summary_rows.append(dict(variant="V4_multiclass", metric="n_detectors",
                              mean=v4_df["n_detectors"].mean(), sd=v4_df["n_detectors"].std(ddof=1)))
    pd.DataFrame(summary_rows).to_csv("results/nslkdd_ablation_summary.csv", index=False)

    # ---- significance: paired tests V0 vs V3, same folds ----
    sig_rows = []
    for metric in ["DR", "FPR", "Acc"]:
        a = np.array(fold_arrays["V0_classical"][metric])
        b = np.array(fold_arrays["V3_full_binary"][metric])
        t_stat, t_p = ttest_rel(b, a)
        try:
            w_stat, w_p = wilcoxon(b, a)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        sig_rows.append(dict(metric=metric, comparison="V3_vs_V0",
                              mean_diff=(b - a).mean(), t_stat=t_stat, t_p=t_p,
                              wilcoxon_stat=w_stat, wilcoxon_p=w_p))
    pd.DataFrame(sig_rows).to_csv("results/nslkdd_significance.csv", index=False)

    # ---- baselines: KNN, KMeans on identical PCA-15 pipeline, single split ----
    print("Running baselines (KNN, KMeans)...")
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG_SEED)
    y_tr_bin = (y_tr != "normal").astype(int)
    y_te_bin = (y_te != "normal").astype(int)

    baseline_rows = []
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr_bin)
    knn_train_t = time.time() - t0
    t0 = time.time()
    pred = knn.predict(X_te)
    knn_infer_t = (time.time() - t0) / len(X_te)
    m = confusion_metrics(y_te_bin.astype(bool), pred.astype(bool))
    baseline_rows.append(dict(model="KNN_reproduced", **m, train_time_s=knn_train_t, infer_time_per_sample_s=knn_infer_t))

    t0 = time.time()
    km = KMeans(n_clusters=2, random_state=RNG_SEED, n_init=10).fit(X_tr)
    km_train_t = time.time() - t0
    # map cluster ids to normal/attack by majority label in each cluster
    cluster_pred_tr = km.predict(X_tr)
    cluster_to_label = {}
    for c in [0, 1]:
        mask = cluster_pred_tr == c
        cluster_to_label[c] = int(np.round(y_tr_bin[mask].mean())) if mask.sum() > 0 else 0
    t0 = time.time()
    cluster_pred_te = km.predict(X_te)
    km_infer_t = (time.time() - t0) / len(X_te)
    pred = np.array([cluster_to_label[c] for c in cluster_pred_te])
    m = confusion_metrics(y_te_bin.astype(bool), pred.astype(bool))
    baseline_rows.append(dict(model="KMeans_reproduced", **m, train_time_s=km_train_t, infer_time_per_sample_s=km_infer_t))

    pd.DataFrame(baseline_rows).to_csv("results/nslkdd_baselines.csv", index=False)

    with open("results/nslkdd_hyperparameters.json", "w") as f:
        json.dump(dict(detector_hp=DETECTOR_HP, n_folds=N_FOLDS, pca_dim=PCA_DIM, rng_seed=RNG_SEED), f, indent=2)

    print("Done. Results written to results/nslkdd_*.csv")


if __name__ == "__main__":
    main()
