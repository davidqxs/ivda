"""Regenerate Table 1: accuracy and cost of the full IVDA at different feature
dimensionalities, so the table is reproducible from the released code.

The original Table 1 was carried over from the lost implementation and could not be
reproduced, which Section 5 had to declare as a limitation. This script re-measures the
same comparison under the pipeline actually used in the paper:

  41   : the 41 original NSL-KDD features, the three categorical ones ordinal-encoded
  122  : one-hot encoding of the categorical features, no PCA
  27/15/11/8 : one-hot encoding followed by PCA to that many components

Everything else is held fixed: the same pooled 148,517 records, the same transforms fitted
on the pooled matrix before splitting (Section 4.1), the same detector hyperparameters, the
same variant (V3, full IVDA) and the same RNG seed. Only the feature space changes, which
is what the table is meant to isolate.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

import sys
sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, NSLKDD_COLUMNS, NSLKDD_CATEGORICAL
from ivda.detector import VDetectorSet
from run_ablation_nslkdd import confusion_metrics, DETECTOR_HP, RNG_SEED, N_FOLDS

# (label, encoding, pca_dim) - label is what appears in Table 1's first column
CONFIGS = [
    ("41", "ordinal", None),
    ("122", "onehot", None),
    ("27", "onehot", 27),
    ("15", "onehot", 15),
    ("11", "onehot", 11),
    ("8", "onehot", 8),
]
FOLDS_PER_CONFIG = 3      # keep the sweep affordable; see caption note


def build(df, encoding, pca_dim):
    feature_cols = [c for c in NSLKDD_COLUMNS if c not in ("label", "difficulty")]
    numeric = [c for c in feature_cols if c not in NSLKDD_CATEGORICAL]
    enc = (OneHotEncoder(handle_unknown="ignore", sparse_output=False) if encoding == "onehot"
           else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    steps = [("pre", ColumnTransformer([("cat", enc, NSLKDD_CATEGORICAL),
                                        ("num", MinMaxScaler(), numeric)]))]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=RNG_SEED)))
    X = Pipeline(steps).fit_transform(df[feature_cols])
    # detector centres are drawn from a bounding box, so the space must be bounded
    X = MinMaxScaler().fit_transform(X)
    return X, df["category"].values


def main():
    print("Loading NSL-KDD...")
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)

    rows = []
    for label, encoding, pca_dim in CONFIGS:
        X, y = build(full, encoding, pca_dim)
        lo, hi = X.min(axis=0), X.max(axis=0)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
        folds = list(skf.split(X, y))[:FOLDS_PER_CONFIG]
        print(f"\n=== {label} features (encoding={encoding}, pca={pca_dim}) dims={X.shape[1]}")

        accs, drs, fprs, dets, t_train, t_inf = [], [], [], [], [], []
        for i, (tr, te) in enumerate(folds):
            self_tr = X[tr][y[tr] == "normal"]
            nonself_tr = X[tr][y[tr] != "normal"]
            y_te = y[te] != "normal"

            t0 = time.time()
            det = VDetectorSet(use_statistical_term=True, use_boundary_aware=True,
                               rng=np.random.default_rng(RNG_SEED * 1000 + i), **DETECTOR_HP)
            det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
            tt = time.time() - t0

            t0 = time.time()
            pred = det.predict_nonself(X[te])
            ti = (time.time() - t0) / len(te)

            m = confusion_metrics(y_te, pred)
            accs.append(m["Acc"]); drs.append(m["DR"]); fprs.append(m["FPR"])
            dets.append(det.n_detectors); t_train.append(tt); t_inf.append(ti)
            print(f"   fold {i}: Acc={m['Acc']:.4f} DR={m['DR']:.4f} FPR={m['FPR']:.4f} "
                  f"n_det={det.n_detectors} t={tt:.1f}s")

        rows.append(dict(features=label, dims=X.shape[1], encoding=encoding,
                         pca_dim=pca_dim if pca_dim else "",
                         Acc_mean=np.mean(accs), Acc_sd=np.std(accs, ddof=1),
                         DR_mean=np.mean(drs), FPR_mean=np.mean(fprs),
                         n_detectors_mean=np.mean(dets),
                         train_time_mean=np.mean(t_train), train_time_sd=np.std(t_train, ddof=1),
                         infer_time_mean=np.mean(t_inf), n_folds=len(folds)))
        pd.DataFrame(rows).to_csv("results/nslkdd_feature_dim.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== Table 1 (regenerated) ===")
    for _, r in df.iterrows():
        print(f"  {r['features']:>4}  Acc={r['Acc_mean']*100:6.2f}±{r['Acc_sd']*100:.2f}  "
              f"train={r['train_time_mean']:6.2f}±{r['train_time_sd']:.2f}s  "
              f"det={r['n_detectors_mean']:.0f}")
    best = df.loc[df["Acc_mean"].idxmax()]
    print(f"\nbest accuracy at {best['features']} features "
          f"({best['Acc_mean']*100:.2f}%)")
    with open("results/nslkdd_feature_dim_meta.json", "w") as f:
        json.dump(dict(folds_per_config=FOLDS_PER_CONFIG, rng_seed=RNG_SEED,
                       variant="V3_full_binary", detector_hp=DETECTOR_HP,
                       best_features=str(best["features"])), f, indent=2)
    print("Done -> results/nslkdd_feature_dim.csv")


if __name__ == "__main__":
    main()
