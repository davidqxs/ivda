"""
Which spike encoding lets a compiled V-detector set run as a spiking network?

run_snn_compile.py established two facts on the canonical split:
  - the compilation is exact: with x~ = [x, ||x||^2] each detector is a linear threshold
    unit, w = [2c, -1], theta = ||c||^2 - r^2, reproducing the V-detector on 99.99% of
    test samples as analog units;
  - rate coding fails: agreement 0.48-0.65 and FPR 35-65% for T in 4..256.

The diagnosed cause is catastrophic cancellation. The decision margin r^2 - ||x - c||^2 is of
order 1e-2, but the expanded form computes it as a difference of terms of order 1-15, with a
compensating weight of p = 15 on the ||x||^2 channel. Any encoding noise on those terms is
amplified past the margin.

This script tests two remedies separately and together, so the effect of each is visible:

  rate            Bernoulli rate coding of x in [0,1] - the failing baseline, re-run here.
  rate_centred    Rate coding after centring at the mean of the normal training samples.
                  Normal traffic is tightly clustered, so centring shrinks every term in the
                  expansion. Signed inputs use an ON/OFF channel pair.
  ttfs_centred    Time-to-first-spike coding after centring: every input emits at most ONE
                  spike, its value carried by spike time, quantised to T levels. Deterministic,
                  so there is no sampling noise, only quantisation. Both the coordinates and
                  the norm channel are quantised.
  ttfs_consistent As ttfs_centred, but the encoder computes the norm channel from the
                  quantised coordinates it emits and sends it at full precision. The decision
                  is then exact for the quantised point, isolating input quantisation as the
                  only error source. (Ablation.)

Centring uses only normal training samples, which every negative-selection method already
consumes as its self set; no attack label is involved.

Energy, analytical 45 nm estimates (Horowitz), reported under two accountings:
  synops     the convention in the SNN literature: spike-triggered synaptic accumulates only,
             SynOps x 0.9 pJ;
  synops+nu  also charging one accumulate per neuron per time step for membrane
             integration - the honest count, which the convention omits.
Dense baseline: the V-detector evaluated as n_det x (p+1) multiply-accumulates at 4.6 pJ.
Neither is a hardware measurement.

Writes results/snn_encodings_canonical.csv.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from data_pipeline import load_nslkdd, build_nslkdd_features
from ivda.detector import VDetectorSet

HP = dict(alpha=0.05, n_probe=3000, c_target=0.95, rho_min=0.0005, boundary_k=30,
          max_detectors=1500, max_consecutive_failures=1500, cap_radius=False, r_min=1e-6)
T_GRID = [4, 8, 16, 32, 64, 128]
E_AC, E_MAC = 0.9, 4.6                      # pJ, 45 nm
ENCODINGS = ["rate", "rate_centred", "ttfs_centred", "ttfs_consistent"]


def decide(Xh, Nh, C, theta, chunk=4000):
    """Any neuron fires: 2 C.x_hat - norm_hat > theta, for each sample."""
    out = np.zeros(len(Xh), dtype=bool)
    for s in range(0, len(Xh), chunk):
        D = 2.0 * (Xh[s:s + chunk] @ C.T) - Nh[s:s + chunk, None]
        out[s:s + chunk] = (D > theta).any(axis=1)
    return out


def encode(X, mu, scale, nscale, enc, T, rng):
    """Return (x_hat, norm_hat, input spikes per sample) under encoding `enc`."""
    Xc = X - mu
    if enc == "rate":
        V = np.clip(Xc / scale, 0.0, 1.0)                   # mu = 0, inputs non-negative
        cnt = rng.binomial(T, V)
        Xh = cnt / T * scale
        u = np.clip((Xc ** 2).sum(axis=1) / nscale, 0.0, 1.0)
        cu = rng.binomial(T, u)
        Nh = cu / T * nscale
        return Xh, Nh, cnt.sum(axis=1) + cu

    V = np.clip(Xc / scale, -1.0, 1.0)                      # signed, ON/OFF pair
    mag, sign = np.abs(V), np.sign(V)
    u = np.clip((Xc ** 2).sum(axis=1) / nscale, 0.0, 1.0)
    if enc == "rate_centred":
        cnt = rng.binomial(T, mag)
        Xh = sign * cnt / T * scale
        cu = rng.binomial(T, u)
        Nh = cu / T * nscale
        return Xh, Nh, cnt.sum(axis=1) + cu

    q = np.round(mag * T) / T                               # TTFS: one spike, time-quantised
    Xh = sign * q * scale
    spikes = (q > 0).sum(axis=1)
    if enc == "ttfs_centred":
        qu = np.round(u * T) / T
        Nh = qu * nscale
        spikes = spikes + (qu > 0)
    elif enc == "ttfs_consistent":
        Nh = (Xh ** 2).sum(axis=1)
        spikes = spikes + (Nh > 0)
    else:
        raise ValueError(enc)
    return Xh, Nh, spikes


def main():
    train, test = load_nslkdd("data/nslkdd/KDDTrain+.txt", "data/nslkdd/KDDTest+.txt")
    X_tr, y_tr, X_te, y_te, _, _ = build_nslkdd_features(train, test, pca_dim=15, random_state=0)
    lo, hi = X_tr.min(axis=0), X_tr.max(axis=0)
    p = X_tr.shape[1]
    S, A = X_tr[y_tr == "normal"], X_tr[y_tr != "normal"]
    attack, normal = y_te != "normal", y_te == "normal"

    mu_c = S.mean(axis=0)                                   # centring point: normal train mean
    rows = []
    for name, flags in [("V3_ivda", dict(use_statistical_term=True, use_boundary_aware=True)),
                        ("V0_classical", dict(use_statistical_term=False, use_boundary_aware=False))]:
        det = VDetectorSet(rng=np.random.default_rng(0), **flags, **HP)
        det.fit(S, lo, hi, non_self_samples=A)
        n = det.n_detectors
        exact = det.predict_nonself(X_te)
        dense_pJ = n * (p + 1) * E_MAC
        print(f"\n### {name}: {n} detectors; dense = {n * (p + 1)} MACs ~ {dense_pJ:.0f} pJ/sample")

        for enc in ENCODINGS:
            centred = enc != "rate"
            mu = mu_c if centred else np.zeros(p)
            C = det.centers_ - mu
            theta = (C ** 2).sum(axis=1) - det.radii_ ** 2
            # per-dimension scale covers the whole box, so no input is ever clipped
            scale = np.maximum(np.abs(lo - mu), np.abs(hi - mu))
            nscale = float((scale ** 2).sum())

            # sanity: the centred compilation must still be exact as analog units
            if enc == ENCODINGS[0] or enc == "rate_centred":
                Xc = X_te - mu
                ana = decide(Xc, (Xc ** 2).sum(axis=1), C, theta)
                print(f"  [{enc:15s}] analog agreement with V-detector: {(ana == exact).mean():.6f}")

            rng = np.random.default_rng(1)
            for T in T_GRID:
                Xh, Nh, spk = encode(X_te, mu, scale, nscale, enc, T, rng)
                pred = decide(Xh, Nh, C, theta)
                synops = spk.mean() * n
                e_syn = synops * E_AC
                e_all = (synops + n * T) * E_AC
                r = dict(variant=name, n_det=n, encoding=enc, T=T,
                         agree=float((pred == exact).mean()),
                         DR=float(pred[attack].mean()), FPR=float(pred[normal].mean()),
                         DR_exact=float(exact[attack].mean()), FPR_exact=float(exact[normal].mean()),
                         spikes=float(spk.mean()), spikes_normal=float(spk[normal].mean()),
                         spikes_attack=float(spk[attack].mean()),
                         synops=synops, E_synops_pJ=e_syn, E_total_pJ=e_all, E_dense_pJ=dense_pJ,
                         ratio_synops=e_syn / dense_pJ, ratio_total=e_all / dense_pJ)
                rows.append(r)
                print(f"  {enc:15s} T={T:4d}  agree={r['agree']:.4f}  DR={r['DR']:.4f}  "
                      f"FPR={r['FPR']:.4f}  spikes={r['spikes']:6.2f}  "
                      f"E/dense: synops {r['ratio_synops']:5.2f}x  total {r['ratio_total']:5.2f}x",
                      flush=True)

    pd.DataFrame(rows).to_csv("results/snn_encodings_canonical.csv", index=False)
    print("\nwrote results/snn_encodings_canonical.csv")


if __name__ == "__main__":
    main()
