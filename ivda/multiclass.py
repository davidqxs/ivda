"""
Multi-class IVDA wrapper, per Section 3.5 of the paper.

For K classes, trains K detector sets D_1..D_K where D_k treats class k as
self and all other classes as non-self (unchanged from the manuscript's
Section 3.5 training definition). Decision rule corrected to argmin (spec
section 4): a sample matched by many D_k detectors is likely NOT class k.
Tie-break by nearest-self-sample distance among tied classes.
"""
import numpy as np
from .detector import VDetectorSet


class MulticlassIVDA:
    def __init__(self, detector_kwargs=None, binary_self_class=None):
        """
        detector_kwargs: passed through to each per-class VDetectorSet.
        binary_self_class: if set (e.g. "normal"), collapses to binary
            classification - self_class vs. everything else - by building
            only D_self and predicting self/non-self from a single set
            (matches the manuscript's binary detection-rate/FPR experiments,
            distinct from the K-way multi-class experiments).
        """
        self.detector_kwargs = detector_kwargs or {}
        self.binary_self_class = binary_self_class
        self.classes_ = None
        self.detectors_ = {}
        self.self_sets_ = {}

    def fit(self, X, y, feature_lo, feature_hi, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        X = np.asarray(X)
        y = np.asarray(y)

        if self.binary_self_class is not None:
            classes = [self.binary_self_class]
        else:
            classes = sorted(np.unique(y).tolist())
        self.classes_ = classes

        for k in classes:
            self_mask = y == k
            non_self_mask = ~self_mask
            self.self_sets_[k] = X[self_mask]
            det = VDetectorSet(rng=np.random.default_rng(rng.integers(1 << 31)), **self.detector_kwargs)
            det.fit(X[self_mask], feature_lo, feature_hi, non_self_samples=X[non_self_mask])
            self.detectors_[k] = det
        return self

    def _phi_matrix(self, X):
        X = np.asarray(X)
        phis = np.column_stack([self.detectors_[k].match_score(X) for k in self.classes_])
        return phis  # (n_samples, n_classes)

    def predict(self, X):
        if self.binary_self_class is not None:
            # binary: non-self if matched by >=1 detector in D_self (self
            # class's detectors cover the *non-self* region directly - this
            # is the classical binary V-detector usage, unchanged from
            # Section 3.3, not the multi-class argmin rule).
            phi = self.detectors_[self.binary_self_class].match_score(X)
            pred_nonself = phi > 0
            return np.where(pred_nonself, "attack", "normal")

        phi = self._phi_matrix(X)  # (n, K)
        y_pred = np.empty(len(X), dtype=object)
        for i in range(len(X)):
            row = phi[i]
            min_val = row.min()
            tied = np.where(row == min_val)[0]
            if len(tied) == 1:
                y_pred[i] = self.classes_[tied[0]]
            else:
                # tie-break: nearest-self-sample distance among tied classes
                x = X[i]
                best_k, best_d = None, np.inf
                for idx in tied:
                    k = self.classes_[idx]
                    d = np.linalg.norm(self.self_sets_[k] - x, axis=1).min()
                    if d < best_d:
                        best_d = d
                        best_k = k
                y_pred[i] = best_k
        return y_pred

    @property
    def n_detectors_total(self):
        return sum(d.n_detectors for d in self.detectors_.values())
