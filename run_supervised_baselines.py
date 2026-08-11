"""Modern supervised baselines (random forest, gradient-boosted trees) reproduced under
the pipeline IVDA itself uses, requested by a co-author in review.

Protocol is deliberately identical to run_ablation_nslkdd.py: the same pooled 148,517
NSL-KDD records, the same PCA-15 feature construction with the same random_state, and the
SAME StratifiedKFold(10, shuffle=True, random_state=0) folds, so these rows are directly
comparable with the IVDA and fixed-radius NSA rows rather than the single 80/20 split used
for the existing KNN and K-means baselines.

Gradient boosting uses scikit-learn's HistGradientBoostingClassifier rather than XGBoost so
that no dependency outside the software stack already declared in Section 4.2 is required.

Also records model size, which is the quantity the paper's compactness argument rests on:
IVDA stores 134 hyperspheres (a centre in R^15 plus a radius); a forest stores every split
node of every tree.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import build_nslkdd_features, load_nslkdd

RNG_SEED = 0
N_FOLDS = 10
PCA_DIM = 15
RESULTS = Path(__file__).parent / "results"


def confusion_metrics(y_true_nonself, y_pred_nonself):
    tp = np.sum(y_pred_nonself & y_true_nonself)
    fn = np.sum(~y_pred_nonself & y_true_nonself)
    fp = np.sum(y_pred_nonself & ~y_true_nonself)
    tn = np.sum(~y_pred_nonself & ~y_true_nonself)
    dr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(DR=dr, FPR=fpr, Acc=acc, TP=tp, FN=fn, FP=fp, TN=tn)


def model_size(name, clf):
    """Stored-parameter count, for the compactness comparison."""
    if name == "RandomForest":
        nodes = int(sum(t.tree_.node_count for t in clf.estimators_))
        return dict(n_trees=len(clf.estimators_), n_nodes=nodes)
    total = 0
    for stage in clf._predictors:
        for pred in stage:
            total += int(pred.nodes.shape[0])
    return dict(n_trees=int(sum(len(s) for s in clf._predictors)), n_nodes=total)


def make(name, fold_i):
    seed = RNG_SEED * 1000 + fold_i
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(random_state=seed)


def main():
    print("Loading NSL-KDD...", flush=True)
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X, y, _, _, _, _ = build_nslkdd_features(
        full, full.iloc[:1], pca_dim=PCA_DIM, random_state=RNG_SEED
    )
    print(f"N={len(X)}, dims={X.shape[1]}", flush=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    folds = list(skf.split(X, y))

    rows = []
    for name in ("RandomForest", "HistGradientBoosting"):
        for fold_i, (tr, te) in enumerate(folds):
            X_tr, X_te = X[tr], X[te]
            y_tr = (y[tr] != "normal").astype(int)
            y_te = (y[te] != "normal").astype(int)

            clf = make(name, fold_i)
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            train_t = time.time() - t0

            t0 = time.time()
            pred = clf.predict(X_te)
            infer_t = (time.time() - t0) / len(X_te)

            m = confusion_metrics(y_te.astype(bool), pred.astype(bool))
            rows.append(dict(model=name, fold=fold_i, **m, train_time_s=train_t,
                             infer_time_per_sample_s=infer_t, **model_size(name, clf)))
            print(f"  {name} fold {fold_i}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} "
                  f"Acc={m['Acc']:.4f} nodes={rows[-1]['n_nodes']} t={train_t:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "nslkdd_supervised_folds.csv", index=False)

    summary = []
    for name, g in df.groupby("model"):
        for metric in ("DR", "FPR", "Acc", "train_time_s", "infer_time_per_sample_s", "n_nodes"):
            summary.append(dict(model=name, metric=metric,
                                mean=g[metric].mean(), sd=g[metric].std(ddof=1)))
    pd.DataFrame(summary).to_csv(RESULTS / "nslkdd_supervised_summary.csv", index=False)

    # paired tests against V3 (full IVDA) on the identical folds
    abl = pd.read_csv(RESULTS / "nslkdd_ablation_folds.csv")
    v3 = abl[abl.variant == "V3_full_binary"].sort_values("fold")
    sig = []
    for name, g in df.groupby("model"):
        g = g.sort_values("fold")
        for metric in ("DR", "FPR", "Acc"):
            a, b = v3[metric].to_numpy(), g[metric].to_numpy()
            t_stat, t_p = ttest_rel(b, a)
            try:
                w_stat, w_p = wilcoxon(b, a)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            sig.append(dict(model=name, metric=metric, comparison=f"{name}_vs_V3",
                            mean_diff=(b - a).mean(), t_stat=t_stat, t_p=t_p,
                            wilcoxon_stat=w_stat, wilcoxon_p=w_p))
    pd.DataFrame(sig).to_csv(RESULTS / "nslkdd_supervised_significance.csv", index=False)

    with open(RESULTS / "nslkdd_supervised_hp.json", "w") as f:
        json.dump(dict(random_forest=dict(n_estimators=100),
                       hist_gradient_boosting="sklearn defaults",
                       n_folds=N_FOLDS, pca_dim=PCA_DIM, rng_seed=RNG_SEED,
                       folds="identical to run_ablation_nslkdd.py"), f, indent=2)

    print("\n=== summary (mean over 10 folds) ===")
    for name, g in df.groupby("model"):
        print(f"  {name}: DR={g.DR.mean()*100:.2f} FPR={g.FPR.mean()*100:.2f} "
              f"Acc={g.Acc.mean()*100:.2f} nodes={g.n_nodes.mean():.0f} "
              f"train={g.train_time_s.mean():.1f}s")
    print("Done -> results/nslkdd_supervised_*.csv")


if __name__ == "__main__":
    main()
