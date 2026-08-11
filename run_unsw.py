"""
Second-dataset experiment (UNSW-NB15), reported in Section 4.10 of the paper -
proposed (V3 full binary / V4 full multiclass) vs. classical V-detector (V0)
only, single stratified 80/20 split (no third-party baselines on this
dataset - plan explicitly descopes that to keep scope bounded).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, ".")
from data_pipeline import load_unsw, build_unsw_features
from ivda.detector import VDetectorSet
from ivda.multiclass import MulticlassIVDA
from run_ablation_nslkdd import confusion_metrics, DETECTOR_HP, RNG_SEED, PCA_DIM


def main():
    print("Loading UNSW-NB15...")
    df = load_unsw("data/unsw/train.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["attack_cat"], random_state=RNG_SEED)
    Xtr, ytr, Xte, yte, pipe, scaler = build_unsw_features(train_df, test_df, pca_dim=PCA_DIM, random_state=RNG_SEED)
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    print(f"N_train={len(Xtr)}, N_test={len(Xte)}, classes={np.unique(ytr)}")

    self_label = "Normal"
    y_tr_bin_self = ytr == self_label
    y_te_nonself = yte != self_label
    self_tr = Xtr[y_tr_bin_self]
    nonself_tr = Xtr[~y_tr_bin_self]

    rows = []
    for name, flags in [
        ("V0_classical", dict(use_statistical_term=False, use_boundary_aware=False)),
        ("V3_full_binary", dict(use_statistical_term=True, use_boundary_aware=True)),
    ]:
        t0 = time.time()
        det = VDetectorSet(rng=np.random.default_rng(RNG_SEED), **flags, **DETECTOR_HP)
        det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
        train_t = time.time() - t0
        t0 = time.time()
        pred = det.predict_nonself(Xte)
        infer_t = (time.time() - t0) / len(Xte)
        m = confusion_metrics(y_te_nonself, pred)
        rows.append(dict(variant=name, **m, n_detectors=det.n_detectors,
                          train_time_s=train_t, infer_time_per_sample_s=infer_t))
        print(f"{name}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} Acc={m['Acc']:.4f} n_det={det.n_detectors} t={train_t:.1f}s")

    pd.DataFrame(rows).to_csv("results/unsw_binary.csv", index=False)

    print("Running UNSW-NB15 multi-class (V4)...")
    t0 = time.time()
    mc = MulticlassIVDA(detector_kwargs=dict(use_statistical_term=True, use_boundary_aware=True, **DETECTOR_HP))
    mc.fit(Xtr, ytr, lo, hi, rng=np.random.default_rng(RNG_SEED))
    train_t = time.time() - t0
    t0 = time.time()
    y_pred = mc.predict(Xte)
    infer_t = (time.time() - t0) / len(Xte)
    acc = np.mean(y_pred == yte)
    print(f"V4_multiclass: Acc={acc:.4f} n_det={mc.n_detectors_total} t={train_t:.1f}s")

    per_class = []
    for cls in mc.classes_:
        mask = yte == cls
        if mask.sum() == 0:
            continue
        tp = np.sum((y_pred == cls) & mask)
        fn = np.sum((y_pred != cls) & mask)
        fp = np.sum((y_pred == cls) & ~mask)
        tn = np.sum((y_pred != cls) & ~mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision and recall) else np.nan
        per_class.append(dict(cls=cls, precision=precision, recall=recall, f1=f1, support=int(mask.sum())))

    pd.DataFrame([dict(variant="V4_multiclass", Acc=acc, n_detectors=mc.n_detectors_total,
                        train_time_s=train_t, infer_time_per_sample_s=infer_t)]).to_csv(
        "results/unsw_multiclass_summary.csv", index=False)
    pd.DataFrame(per_class).to_csv("results/unsw_per_class.csv", index=False)

    with open("results/unsw_hyperparameters.json", "w") as f:
        json.dump(dict(detector_hp=DETECTOR_HP, pca_dim=PCA_DIM, rng_seed=RNG_SEED, self_label=self_label), f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
