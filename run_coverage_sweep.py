"""
Regenerates Figure 3's underlying data: DR/FPR/Acc/detector-count vs. target
coverage C_target, for the full IVDA (V3, binary) on NSL-KDD, single
standard split (fast enough to sweep several C_target values; the 10-fold
ablation in Section 4.2 is the higher-confidence result - this sweep exists
specifically to give Figure 3 a real, regenerated basis instead of the lost
original sweep data).
"""
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet
from run_ablation_nslkdd import confusion_metrics, DETECTOR_HP, RNG_SEED, PCA_DIM

C_TARGETS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    full = pd.concat([train, test], ignore_index=True)
    X_all, y_all, _, _, pipe, scaler = build_nslkdd_features(full, full.iloc[:1], pca_dim=PCA_DIM, random_state=RNG_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=RNG_SEED)
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    self_tr = X_tr[y_tr == "normal"]
    nonself_tr = X_tr[y_tr != "normal"]
    y_te_nonself = y_te != "normal"

    rows = []
    hp = dict(DETECTOR_HP)
    for ct in C_TARGETS:
        hp["c_target"] = ct
        t0 = time.time()
        det = VDetectorSet(use_statistical_term=True, use_boundary_aware=True,
                            rng=np.random.default_rng(RNG_SEED), **hp)
        det.fit(self_tr, lo, hi, non_self_samples=nonself_tr)
        train_t = time.time() - t0
        pred = det.predict_nonself(X_te)
        m = confusion_metrics(y_te_nonself, pred)
        rows.append(dict(c_target=ct, **m, n_detectors=det.n_detectors, train_time_s=train_t))
        print(f"C_target={ct}: DR={m['DR']:.4f} FPR={m['FPR']:.4f} Acc={m['Acc']:.4f} "
              f"n_det={det.n_detectors} t={train_t:.1f}s")

    pd.DataFrame(rows).to_csv("results/nslkdd_coverage_sweep.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
