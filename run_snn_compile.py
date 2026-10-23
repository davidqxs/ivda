"""
Compiling a V-detector set into a single-layer spiking network.

A V-detector matches x when ||x - c||^2 < r^2. Expanding the square,

    ||x - c||^2 < r^2   <=>   2 c.x - ||x||^2  >  ||c||^2 - r^2 .

With the augmented input  x~ = [x, ||x||^2]  every detector is therefore exactly a linear
threshold unit, with weights w = [2c, -1] and threshold theta = ||c||^2 - r^2, both written
down analytically from the detector's centre and radius - no training and no conversion
loss. A detector set is then a single layer of integrate-and-fire neurons, and "matched by
any detector" is "any neuron fires".

This script checks three things on the canonical KDDTrain+/KDDTest+ split:

 1. Exactness of the algebra: the analog threshold units must reproduce the V-detector's
    decisions on every test sample.
 2. Rate-coded spiking inference: each input feature in [0, 1] drives a Bernoulli spike
    train over T time steps, the neurons integrate without leak, and a neuron fires if its
    membrane is above threshold at the end of the window. For a non-leaky integrator read
    out at the end of the window only the per-input spike counts matter, so those are drawn
    directly as Binomial(T, rate) - equivalent, and far cheaper than simulating each step.
    Reported against T: agreement with the exact V-detector, DR and FPR.
 3. An energy estimate from the standard 45 nm figures used throughout the SNN literature
    (Horowitz): 0.9 pJ per accumulate for a spike-triggered synaptic operation, 4.6 pJ per
    multiply-accumulate for the dense computation. These are analytical estimates, not
    hardware measurements, and are reported as such.

The ||x||^2 feature lies in [0, p] for x in [0, 1]^p, so it is fed at rate ||x||^2 / p with
its weight scaled by p to compensate.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet

HP = dict(alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
          max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6)
T_GRID = [4, 8, 16, 32, 64, 128, 256]
E_AC, E_MAC = 0.9e-12, 4.6e-12        # J per accumulate / multiply-accumulate, 45 nm


def compile_detectors(centers, radii, p):
    """Weights (n_det, p+1) and thresholds (n_det,) on the rate-scaled augmented input."""
    W = np.hstack([2.0 * centers, -np.full((len(centers), 1), float(p))])
    theta = (centers ** 2).sum(axis=1) - radii ** 2
    return W, theta


def augment(X):
    p = X.shape[1]
    return np.hstack([X, (X ** 2).sum(axis=1, keepdims=True) / p])


def any_fire(U, W, theta, chunk=4000):
    """Exact analog readout: does any unit exceed threshold?"""
    out = np.zeros(len(U), dtype=bool)
    for s in range(0, len(U), chunk):
        out[s:s + chunk] = ((U[s:s + chunk] @ W.T) > theta).any(axis=1)
    return out


def spiking_fire(U, W, theta, T, rng, chunk=4000):
    """Rate-coded, non-leaky IF neurons read out at the end of a T-step window."""
    out = np.zeros(len(U), dtype=bool)
    in_spikes = np.zeros(len(U))
    for s in range(0, len(U), chunk):
        counts = rng.binomial(T, np.clip(U[s:s + chunk], 0.0, 1.0))
        in_spikes[s:s + chunk] = counts.sum(axis=1)
        out[s:s + chunk] = (((counts / T) @ W.T) > theta).any(axis=1)
    return out, in_spikes


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    X_tr, y_tr, X_te, y_te, _, _ = build_nslkdd_features(train, test, pca_dim=15, random_state=0)
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    p = X_tr.shape[1]
    S, A = X_tr[y_tr == "normal"], X_tr[y_tr != "normal"]
    attack, normal = y_te != "normal", y_te == "normal"
    U = augment(X_te)

    rows = []
    for name, flags in [("V3_ivda", dict(use_statistical_term=True, use_boundary_aware=True)),
                        ("V0_classical", dict(use_statistical_term=False, use_boundary_aware=False))]:
        det = VDetectorSet(rng=np.random.default_rng(0), **flags, **HP)
        det.fit(S, lo, hi, non_self_samples=A)
        n = det.n_detectors
        exact = det.predict_nonself(X_te)
        W, theta = compile_detectors(det.centers_, det.radii_, p)
        analog = any_fire(U, W, theta)
        agree_analog = float((analog == exact).mean())
        print(f"\n### {name}: {n} detectors -> {n} spiking neurons, {p + 1} inputs")
        print(f"  [1] analog threshold units vs exact V-detector: agreement {agree_analog:.6f} "
              f"({int((analog != exact).sum())} of {len(exact)} samples differ)")

        dense_E = n * (p + 1) * E_MAC
        rows.append(dict(variant=name, n_det=n, T="exact", agree=1.0,
                         DR=float(exact[attack].mean()), FPR=float(exact[normal].mean()),
                         synops=np.nan, energy_pJ=dense_E * 1e12, vs_dense=1.0))
        print(f"  dense inference : {n * (p + 1)} MACs  ~ {dense_E * 1e12:9.1f} pJ/sample")

        rng = np.random.default_rng(1)
        for T in T_GRID:
            spk, in_spikes = spiking_fire(U, W, theta, T, rng)
            synops = in_spikes.mean() * n          # each input spike fans out to every neuron
            E = synops * E_AC
            rows.append(dict(variant=name, n_det=n, T=T, agree=float((spk == exact).mean()),
                             DR=float(spk[attack].mean()), FPR=float(spk[normal].mean()),
                             synops=synops, energy_pJ=E * 1e12, vs_dense=E / dense_E))
            print(f"  T={T:4d}  agree={rows[-1]['agree']:.4f}  DR={rows[-1]['DR']:.4f}  "
                  f"FPR={rows[-1]['FPR']:.4f}  SynOps={synops:9.0f}  ~{E * 1e12:9.1f} pJ  "
                  f"({E / dense_E:5.2f}x dense)")

    pd.DataFrame(rows).to_csv("results/snn_compile_canonical.csv", index=False)
    print("\nwrote results/snn_compile_canonical.csv")


if __name__ == "__main__":
    main()
